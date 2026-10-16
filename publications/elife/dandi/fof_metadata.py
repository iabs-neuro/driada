"""Descriptive metadata of the FOF dataset written into every NWB file.

Everything here is taken from the Methods of the DRIADA paper (animals,
surgery, behavioural procedure, behavioural analysis, calcium processing) or
measured on the synchronized files (units, zone relations). Per-animal and
per-session values live in ``subjects.csv`` and ``sessions.csv``.
"""

PROTOCOL = "FOF"

INSTITUTION = "Institute for Advanced Brain Studies, Lomonosov Moscow State University"

# The paper names no laboratory, so the NWB ``lab`` field is left empty.
LAB = None

# Authors with the "Investigation" role in the author contributions of the
# paper, in the "Last, First" form NWB expects.
EXPERIMENTERS = [
    "Rogozhnikova, Olga",
    "Ivashkina, Olga",
    "Toropova, Ksenia",
    "Ivanova, Anna",
]

KEYWORDS = [
    "hippocampus",
    "CA1",
    "calcium imaging",
    "one-photon miniscope",
    "GCaMP6s",
    "open field",
    "spatial navigation",
    "place cells",
    "head direction",
    "behavior",
]

EXPERIMENT_DESCRIPTION = (
    "Free exploration of a circular open-field arena (diameter 96 cm) with visual "
    "landmarks on the surrounding walls: one session of about 12 minutes per day on "
    "three consecutive days per mouse. Calcium activity of dorsal CA1 was recorded "
    "with a head-mounted Miniscope V4.4 while behaviour was filmed by a top-mounted "
    "camera; the two streams were synchronized in Bonsai using shared timestamps. "
    "The file holds processed data only: fluorescence traces of the retained CNMF-E "
    "components and behavioural variables on the time base of the imaging frames "
    "(nominal 30 frames per second). Raw miniscope and behaviour videos, spatial "
    "footprints of the components and their positions in the field of view are not "
    "included."
)

SESSION_DESCRIPTION = (
    "Open-field exploration, day {day_number} of 3, mouse {subject_id}: one-photon "
    "calcium imaging of dorsal CA1 with synchronized behavioural variables."
)

SUBJECT_DESCRIPTION = (
    "C57BL/6J mouse with a GRIN lens implanted above dorsal CA1 and a baseplate for "
    "a Miniscope V4.4. Age is the exact age on the day of this session."
)

VIRUS = (
    "AAV/DJ-CAG-GCaMP6s, 300 nL, unilateral injection into dorsal CA1 "
    "(AP -1.94 mm, ML +1.46 mm, DV -1.2 mm)."
)

SURGERY = (
    "Three stages one week apart: viral injection, implantation of a 1-mm-diameter "
    "GRIN lens 200 um above CA1, installation of an aluminum baseplate for miniscope "
    "mounting. Anesthesia: zoletil (40 mg/kg) and xylazine (5 mg/kg), "
    "intraperitoneal; dexamethasone (4 mg/kg) subcutaneously before surgery."
)

DEVICE_NAME = "UCLA Miniscope v4.4"
DEVICE_DESCRIPTION = "Head-mounted one-photon miniature microscope (Miniscope V4.4)."
INDICATOR = "GCaMP6s"
# The archive accepts only a name or an abbreviation of the Allen Mouse Brain
# atlas here; that the dorsal part was imaged is said in the plane description.
LOCATION = "CA1"
IMAGING_PLANE_DESCRIPTION = (
    "One-photon calcium imaging plane in dorsal hippocampal CA1 of the mouse, "
    "recorded through a GRIN lens with a Miniscope V4.4. Viral expression: "
    "AAV/DJ-CAG-GCaMP6s. Stereotaxic target of the injection: AP -1.94 mm, "
    "ML +1.46 mm, DV -1.2 mm."
)

CALCIUM_DESCRIPTION = (
    "Fluorescence traces (dF/F) of the retained CNMF-E components, produced by the "
    "BEARMIND pipeline (motion correction with NoRMCorre, source extraction with "
    "CNMF-E, visual inspection of the components) and aligned to behaviour. "
    "Rows are imaging frames, columns are components. Values are stored exactly as "
    "they enter the analysis of the paper, negative values included."
)

ROI_TABLE_DESCRIPTION = (
    "One row per retained CNMF-E component, in the column order of the Calcium "
    "series. Spatial footprints and centre coordinates of the components are NOT "
    "part of this dataset: image_mask is a 1x1 zero placeholder that the table "
    "format requires and carries no information."
)

IMAGE_MASK_DESCRIPTION = (
    "Placeholder (1x1 zeros): no spatial footprints are available in this dataset."
)

POSITION_UNIT = "centimeters"
POSITION_DESCRIPTION = (
    "Position of the body centre (columns: x, y), tracked with DeepLabCut and "
    "processed with the Sphynx package, on the time base of the imaging frames."
)
POSITION_REFERENCE_FRAME = (
    "Coordinates in the plane of the arena floor, in centimetres, in the frame of "
    "the top-mounted behaviour camera. The origin is a corner of that frame, not "
    "the centre of the arena."
)

_ANGLE_FRAME = (
    "Angle in the x-y plane of the Position series, atan2(dy, dx) convention: "
    "0 along +x, pi/2 along +y."
)
_STATE_NOTE = (
    " 1 while the state holds, 0 otherwise; values between 0 and 1 occur at state "
    "boundaries and come from resampling to the imaging frames."
)

# name -> (series type, unit, description, reference frame or None).
# Directions are angles, the one kind of non-position variable SpatialSeries is
# meant for, and take radians. Speed and the state indicators are not spatial
# quantities: SpatialSeries would force a spatial unit on them, so they are
# plain TimeSeries with their real unit (cm/s) or none (dimensionless).
BEHAVIOR_SERIES = {
    "speed": ("TimeSeries", "cm/s",
              "Absolute velocity of the body centre, smoothed with a 0.25-second window.",
              None),
    "bodydirection": ("SpatialSeries", "radians",
                      "Body direction: angle from the body centre to the head centre.",
                      _ANGLE_FRAME),
    "headdirection": ("SpatialSeries", "radians",
                      "Head direction: angle from the head centre to the nose tip. "
                      "Stored as in the source; a small share of samples lies outside "
                      "[-pi, pi].",
                      _ANGLE_FRAME),
    "rest": ("TimeSeries", "n.a.",
             "Rest: speed below 1 cm/s." + _STATE_NOTE, None),
    "walk": ("TimeSeries", "n.a.",
             "Slow locomotion (walk): speed between 1 and 5 cm/s." + _STATE_NOTE, None),
    "locomotion": ("TimeSeries", "n.a.",
                   "Fast locomotion (run): speed above 5 cm/s." + _STATE_NOTE, None),
    "freezing": ("TimeSeries", "n.a.",
                 "Freezing: body centre velocity below 1 cm/s together with nose tip "
                 "velocity below 2 cm/s." + _STATE_NOTE, None),
    "rear": ("TimeSeries", "n.a.",
             "Rearing, detected by a transient reduction of the distance between the "
             "hind limbs and the tail base." + _STATE_NOTE, None),
    "walls": ("TimeSeries", "n.a.",
              "Body centre in the wall zone, a 10-cm-wide band along the perimeter of "
              "the arena." + _STATE_NOTE, None),
    "centermiddle": ("TimeSeries", "n.a.",
                     "Body centre in the intermediate ring, 20 cm wide, between the "
                     "wall zone and the central zone." + _STATE_NOTE, None),
    "centertrue": ("TimeSeries", "n.a.",
                   "Body centre in the central zone, a circle 40 cm in diameter."
                   + _STATE_NOTE, None),
    "center": ("TimeSeries", "n.a.",
               "Body centre outside the wall zone (intermediate ring or central zone)."
               + _STATE_NOTE, None),
}
