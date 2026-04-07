import numpy as np
import sys
import uuid

import json
from pathlib import Path
from datetime import datetime, timezone

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import Fluorescence, RoiResponseSeries, ImageSegmentation, OpticalChannel
from pynwb.behavior import Position, SpatialSeries

from driada.experiment.exp_build import load_exp_from_aligned_data


def save_exp_to_nwb(
        exp,
        output_path,
        session_description='Driada Experiment: Calcium imaging + Behavior',
        identifier=None,
        session_start_time=None,
        protocol=None,
        device_name='Microscope',
        optical_channel_name='Ch1',
        optical_channel_desc='525nm',
        emission_lambda=525.0,
        imaging_plane_name='ImagingPlane',
        imaging_plane_desc='Brain Region',
        excitation_lambda=488.0,
        indicator='GCaMP',
        location='TargetArea',
        pos_reference_frame='arena_center',
        feat_reference_frame='none',
        pos_unit='pixels',
        feat_unit='various',
        verbose=False
):
    """
    Save a DRIADA experiment object to a standardized NWB file.

    Parameters
    ----------
    exp : driada.experiment.Experiment
        The experiment object containing calcium data, behavior, and metadata.
    output_path : str or Path
        Directory where the resulting NWB file will be saved.
    session_description : str, optional
        Brief description of the experimental session.
    identifier : str, optional
        Unique identifier for the session. If None, a random UUID is generated.
    session_start_time : datetime, optional
        Timestamp of session start. If None, extracted from exp metadata.
    protocol : str, optional
        Experimental protocol name. If None, extracted from exp metadata.
    device_name : str, optional
        Name of the recording device (microscope).
    optical_channel_name : str, optional
        Name of the optical channel.
    optical_channel_desc : str, optional
        Description/filter info for the optical channel.
    emission_lambda : float, optional
        Emission wavelength in nm.
    imaging_plane_name : str, optional
        Name of the imaging plane.
    imaging_plane_desc : str, optional
        Description of the brain region or imaging plane.
    excitation_lambda : float, optional
        Excitation wavelength in nm.
    indicator : str, optional
        Calcium indicator used (e.g., 'GCaMP', 'OGB-1').
    location : str, optional
        Anatomical location (e.g., 'TargetArea', 'CA1').
    pos_reference_frame : str, optional
        Reference frame for the Position SpatialSeries.
    feat_reference_frame : str, optional
        Reference frame for other behavioral SpatialSeries.
    pos_unit : str, optional
        Units for position data (e.g., 'pixels', 'meters').
    feat_unit : str, optional
        Units for other behavioral features.
    verbose : bool, default False
        If True, prints progress and success messages to the console.

    Returns
    -------
    None
    """
    meta = exp.metadata or {}
    fps = float(exp.fps)
    session_name = meta.get('session_name', 'unknown_session')

    if identifier is None:
        identifier = str(uuid.uuid4())

    if session_start_time is None:
        session_start_time = datetime.fromisoformat(
            meta.get('export_timestamp', datetime.now().isoformat())
        ).astimezone(timezone.utc)

    if protocol is None:
        protocol = meta.get('track')

    if verbose:
        print(f"Saving experiment: {session_name} to NWB")

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        session_id=session_name,
        protocol=protocol
    )

    parts = session_name.split('_')
    animal_id = parts[1] if len(parts) > 1 else "unknown"
    notes_str = f"animal_id: {animal_id}"

    extra_meta = {k: v for k, v in meta.items() if k not in ['metrics_df', 'session_name', 'fps', 'export_timestamp']}

    if extra_meta:
        meta_json = json.dumps(extra_meta, default=str, indent=2)
        notes_str += f"\n\n--- DRIADA METADATA ---\n{meta_json}"
    nwbfile.notes = notes_str

    device = nwbfile.create_device(name=device_name)
    optical_channel = OpticalChannel(
        name=optical_channel_name,
        description=optical_channel_desc,
        emission_lambda=emission_lambda
    )
    imaging_plane = nwbfile.create_imaging_plane(
        name=imaging_plane_name,
        optical_channel=optical_channel,
        device=device,
        description=imaging_plane_desc,
        excitation_lambda=excitation_lambda,
        indicator=indicator,
        location=location
    )

    ophys_mod = nwbfile.create_processing_module('ophys', 'Calcium imaging data')
    img_seg = ImageSegmentation()
    ophys_mod.add(img_seg)
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation', description='ROIs', imaging_plane=imaging_plane
    )

    num_neurons = exp.n_cells
    metrics_df = meta.get('metrics_df', {})

    valid_indices = list(range(num_neurons))
    if 'decision' in metrics_df:
        ok_indices = [i for i, d in enumerate(metrics_df['decision']) if d == 'ok']
        if len(ok_indices) == num_neurons:
            valid_indices = ok_indices

    custom_cols = []
    for key, vals in metrics_df.items():
        if key not in ['center', 'decision', 'component_idx'] and isinstance(vals, list):
            ps.add_column(name=key, description=f'DRIADA metric: {key}')
            custom_cols.append(key)

    centers = metrics_df.get('center', [])
    for idx in valid_indices:
        if centers and idx < len(centers):
            y, x = centers[idx]
            mask = [(int(x), int(y), 1.0)]
        else:
            mask = [(0, 0, 1.0)]

        roi_kwargs = {'pixel_mask': mask}
        for col in custom_cols:
            val = metrics_df[col][idx]
            roi_kwargs[col] = np.nan if val is None else val

        ps.add_roi(**roi_kwargs)

    rt_region = ps.create_roi_table_region(description='All ROIs', region=list(range(num_neurons)))

    calcium_series = RoiResponseSeries(
        name='Calcium', data=exp.calcium.data.T, rois=rt_region, unit='df/f', rate=fps
    )
    series_list = [calcium_series]

    if hasattr(exp, 'reconstructions') and exp.reconstructions is not None:
        series_list.append(RoiResponseSeries(
            name='Reconstructions', data=exp.reconstructions.data.T, rois=rt_region, unit='df/f', rate=fps
        ))

    ophys_mod.add(Fluorescence(roi_response_series=series_list))

    beh_mod = nwbfile.create_processing_module('behavior', 'Tracking and state masks')

    x_data = exp.dynamic_features['x'].data
    y_data = exp.dynamic_features['y'].data

    pos = Position()
    pos.create_spatial_series(
        name='SpatialSeries', data=np.vstack([x_data, y_data]).T,
        reference_frame=pos_reference_frame, rate=fps, unit=pos_unit
    )
    beh_mod.add(pos)

    exclude = {'x', 'y'}

    for feature_name in exp.dynamic_features.keys():
        if feature_name not in exclude and not feature_name.endswith('_2d'):
            feat = exp.dynamic_features[feature_name]
            s_ts = SpatialSeries(
                name=feature_name,
                data=feat.data,
                reference_frame=feat_reference_frame,
                rate=fps,
                unit=feat_unit,
                description=f'Behavioral feature: {feature_name}'
            )
            beh_mod.add(s_ts)

    final_output = Path(output_path) / f"{session_name}.nwb"
    with NWBHDF5IO(str(final_output), 'w') as io:
        io.write(nwbfile)

    if verbose:
        print(f"Experiment saved: {final_output.name}")

    return None


def load_exp_from_nwb(
        path,
        data_source='IABS_NWB',
        save_npz=False,
        verbose=False
):
    """
    Load an NWB file and reconstruct a DRIADA experiment object.

    Parameters
    ----------
    path : str or Path
        Path to the source NWB file.
    data_source : str, optional
        Identifier for the data source in DRIADA.
    save_npz : bool, default False
        If True, exports the loaded data into a lightweight NPZ file.
    verbose : bool, default False
        If True, prints progress and cache messages to the console.

    Returns
    -------
    driada.experiment.Experiment
        A fully initialized experiment object ready for DRIADA analysis.
    """
    path = Path(path)
    if verbose:
        print(f"Loading file: {path.name}")

    sys.modules['numpy._core'] = np

    with NWBHDF5IO(str(path), mode='r') as io:
        nwb = io.read()

        notes = nwb.notes or ""
        animal_id = "Unknown"
        if "animal_id:" in notes:
            first_line = notes.split('\n')[0]
            animal_id = first_line.replace('animal_id: ', '').strip()

            extra_metadata = {}
            if "--- DRIADA METADATA ---" in notes:
                try:
                    json_str = notes.split("--- DRIADA METADATA ---")[1].strip()
                    extra_metadata = json.loads(json_str)
                except Exception:
                    pass

        extra_metadata['session_name'] = nwb.session_id

        exp_params = {
            'track': nwb.protocol,
            'animal_id': animal_id,
            'session': nwb.session_id
        }

        ophys_mod = nwb.processing['ophys']
        fluo = ophys_mod.data_interfaces['Fluorescence']
        series = fluo.roi_response_series['Calcium']

        metrics_df = {}
        if 'ImageSegmentation' in ophys_mod.data_interfaces:
            ps = ophys_mod.data_interfaces['ImageSegmentation']['PlaneSegmentation']

            for col_name in ps.colnames:
                if col_name != 'pixel_mask':
                    metrics_df[col_name] = ps[col_name].data[:].tolist()

            centers = []
            for i in range(len(ps)):
                mask = ps['pixel_mask'][i]
                centers.append([mask[0][1], mask[0][0]])
            metrics_df['center'] = centers

        data = {'Calcium': series.data[:].T, '_metadata': {**extra_metadata, 'metrics_df': metrics_df}}

        if 'Reconstructions' in fluo.roi_response_series:
            data['Reconstructions'] = fluo.roi_response_series['Reconstructions'].data[:].T

        if 'behavior' in nwb.processing:
            beh_mod = nwb.processing['behavior']
            for name, interface in beh_mod.data_interfaces.items():
                if name == 'Position':
                    pos_series = interface.spatial_series['SpatialSeries']
                    pos_data = pos_series.data[:]
                    data['x'] = pos_data[:, 0]
                    data['y'] = pos_data[:, 1]
                else:
                    if hasattr(interface, 'data'):
                        data[name] = interface.data[:]

        if save_npz:
            npz_output = path.with_suffix('.npz')
            save_meta = {
                'fps': series.rate,
                'session_name': nwb.session_id,
                'animal_id': animal_id,
                'export_timestamp': nwb.session_start_time.isoformat()
            }
            np.savez(npz_output, _metadata=save_meta, **data)
            if verbose:
                print(f"Data exported to {npz_output.name}")

        return load_exp_from_aligned_data(
            data_source=data_source,
            exp_params=exp_params,
            data=data,
            static_features={'fps': series.rate},
            verbose=verbose
        )