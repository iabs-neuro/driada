import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import Fluorescence, RoiResponseSeries, ImageSegmentation, OpticalChannel
from pynwb.behavior import Position, SpatialSeries

from driada.experiment.exp_build import load_exp_from_aligned_data
from driada.utils.naming import parse_iabs_filename


def save_exp_to_nwb(
        exp,
        output_path,
        session_description='Driada Experiment: Calcium imaging + Behavior',
        identifier=None,
        session_start_time=None,
        protocol=None,
        device_name='UCLA Miniscope v4.4',
        optical_channel_name='GCaMP_channel',
        optical_channel_desc='Single green fluorescence channel '
                             'for one-photon calcium imaging with Miniscope v4.4 '
                             '(GCaMP6s-compatible filter set).',
        emission_lambda=525.0,
        imaging_plane_name='CA1_imaging_plane',
        imaging_plane_desc='One-photon calcium imaging plane recorded from '
                           'dorsal hippocampal CA1 in mouse using UCLA Miniscope v4.4. '
                           'Viral expression: AAV/DJ-CAG-GCaMP6s. '
                           'Stereotaxic target coordinates: AP -1.94 mm, ML 1.46 mm, '
                           'DV 1.20 mm relative to bregma.',
        excitation_lambda=470.0,
        indicator='GCaMP6s',
        location='Hippocampus, dorsal CA1',
        pos_reference_frame='ROI coordinates are defined relative to the top-left corner '
                            'of the motion-corrected imaging field of view '
                            '(image origin: x increases to the right, y increases downward).',
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

    parsed = parse_iabs_filename(session_name)
    animal_id = parsed['animal_id'] if parsed else "Unknown"
    # `notes` is reserved for free-form experimenter text; structured
    # metadata is stored separately in NWB scratch.
    nwbfile.notes = f"animal_id: {animal_id}"

    extra_meta = {
        k: v for k, v in meta.items()
        if k not in ['metrics_df', 'session_name', 'fps', 'export_timestamp']
    }
    if extra_meta:
        nwbfile.add_scratch(
            name='driada_metadata',
            data=json.dumps(extra_meta, default=str, indent=2),
            description='DRIADA-specific experiment metadata (JSON-encoded)',
        )

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

    # All exported cells must have decision='ok'; rejected cells are
    # expected to be filtered out upstream of this function.
    if 'decision' in metrics_df:
        bad = [i for i, d in enumerate(metrics_df['decision']) if d != 'ok']
        if bad:
            preview = bad[:5]
            suffix = '...' if len(bad) > 5 else ''
            raise ValueError(
                f"save_exp_to_nwb: expected all cells to have decision='ok', "
                f"but {len(bad)} cells are rejected (indices: {preview}{suffix}). "
                f"Filter rejected cells before saving."
            )

    # DRIADA tracks only neuron centers, not spatial footprints. Centers
    # are stored as explicit scalar columns; image_mask is left as a 1x1
    # null placeholder because pynwb requires at least one mask per ROI.
    ps.add_column(name='center_y', description='Neuron center row (pixels)')
    ps.add_column(name='center_x', description='Neuron center column (pixels)')

    reserved_cols = {'center', 'decision', 'component_idx', 'center_y', 'center_x'}
    custom_cols = []
    for key, vals in metrics_df.items():
        if key in reserved_cols:
            continue
        if isinstance(vals, list):
            ps.add_column(name=key, description=f'DRIADA metric: {key}')
            custom_cols.append(key)

    centers = metrics_df.get('center', [])
    null_image_mask = np.zeros((1, 1), dtype=np.uint8)
    for idx in range(num_neurons):
        if centers and idx < len(centers):
            y_val, x_val = centers[idx]
            cy = float(y_val) if y_val is not None else float('nan')
            cx = float(x_val) if x_val is not None else float('nan')
        else:
            cy = float('nan')
            cx = float('nan')

        roi_kwargs = {
            'image_mask': null_image_mask,
            'center_y': cy,
            'center_x': cx,
        }
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

    beh_mod = nwbfile.create_processing_module('behavior', 'Tracking position and behavioral features')

    x_data = exp.dynamic_features['x'].data
    y_data = exp.dynamic_features['y'].data

    pos = Position()
    pos.create_spatial_series(
        name='SpatialSeries', data=np.vstack([x_data, y_data]).T,
        reference_frame=pos_reference_frame, rate=fps, unit=pos_unit
    )
    beh_mod.add(pos)

    exclude = {'x', 'y'}
    feature_set = set(exp.dynamic_features.keys())

    for feature_name in exp.dynamic_features.keys():
        if feature_name in exclude:
            continue

        if feature_name.endswith('_2d'):
            parent = feature_name[:-3]
            if parent in feature_set:
                # _2d is a cos/sin projection of `parent` and will be rebuilt
                # on load via create_circular_2d=True. Skip to keep the NWB
                # free of redundant derived data.
                if verbose:
                    print(
                        f"  skipping derived '{feature_name}' "
                        f"(will be rebuilt from '{parent}' on load)"
                    )
            else:
                warnings.warn(
                    f"Dropping derived feature '{feature_name}' with no 1-d "
                    f"parent '{parent}' present in the experiment. It cannot "
                    f"be rebuilt on load and will be permanently lost from "
                    f"the NWB file. Persist '{parent}' alongside "
                    f"'{feature_name}' to preserve it.",
                    UserWarning,
                    stacklevel=2,
                )
            continue

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

    with NWBHDF5IO(str(path), mode='r') as io:
        nwb = io.read()

        notes = nwb.notes or ""
        animal_id = "Unknown"
        extra_metadata = {}

        if "animal_id:" in notes:
            first_line = notes.split('\n')[0]
            animal_id = first_line.replace('animal_id: ', '').strip()

        if 'driada_metadata' in nwb.scratch:
            raw = nwb.scratch['driada_metadata'].data
            if hasattr(raw, 'item'):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            extra_metadata = json.loads(raw)

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

            skip_cols = {
                'pixel_mask', 'voxel_mask', 'image_mask',
                'center_y', 'center_x',
            }
            for col_name in ps.colnames:
                if col_name in skip_cols:
                    continue
                metrics_df[col_name] = ps[col_name].data[:].tolist()

            if 'center_y' in ps.colnames and 'center_x' in ps.colnames:
                ys = ps['center_y'].data[:].tolist()
                xs = ps['center_x'].data[:].tolist()
                metrics_df['center'] = [[y, x] for y, x in zip(ys, xs)]

        meta_out = dict(extra_metadata)
        if metrics_df:
            meta_out['metrics_df'] = metrics_df
        data = {'Calcium': series.data[:].T, '_metadata': meta_out}

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