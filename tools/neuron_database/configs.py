"""Experiment configurations for neuron_database.

Each config bundles session names, exclusions, matching info, and aggregate
features so that loading an experiment is a one-liner.
"""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Configuration for a cross-session INTENSE analysis experiment."""
    experiment_id: str
    sessions: list
    matching_subdir: str = 'Matching'
    tables_subdir: str = 'tables_disentangled'
    nontrivial_matching: bool = True
    delay_strategy: str = 'nonnegative'
    sessions_to_match: list = None
    mice_metadata: list = field(default_factory=list)
    killed_sessions: list = field(default_factory=list)
    excluded_mice: list = field(default_factory=list)
    aggregate_features: dict = field(default_factory=dict)


DELAY_STRATEGY = 'all'

_NOF_AGGREGATE = {'any object': ['object1', 'object2', 'object3', 'object4', 'objects']}

EXPERIMENT_CONFIGS = {
    'NOF': ExperimentConfig(
        experiment_id='NOF',
        sessions=['1D', '2D', '3D', '4D'],
        delay_strategy=DELAY_STRATEGY,
        aggregate_features=_NOF_AGGREGATE,
    ),

    'LNOF': ExperimentConfig(
        experiment_id='LNOF',
        sessions=['1D', '2D', '3D', '4D'],
        matching_subdir='MatchData',
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Group'],
        excluded_mice=['J20', 'J21'],
        aggregate_features=_NOF_AGGREGATE,
    ),

    'LNOF10': ExperimentConfig(
        experiment_id='LNOF10',
        sessions=['1D', '2D', '3D', '4D'],
        matching_subdir='MatchData',
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Group'],
        excluded_mice=['J20', 'J21'],
        aggregate_features=_NOF_AGGREGATE,
    ),

    'BOWL': ExperimentConfig(
        experiment_id='BOWL',
        sessions=['1D', '2D', '3D', '4D', '5D'],
        matching_subdir='MatchData',
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Line', 'Group'],
        killed_sessions=['BOWL_F30_3D'],
    ),

    '3DM': ExperimentConfig(
        experiment_id='3DM',
        sessions=['1D', '2D', '3D', '4D', '5D', '6D', '7D'],
        matching_subdir='MatchData',
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Group'],
        killed_sessions=[
            '3DM_F36_2D', '3DM_F36_6D', '3DM_F38_2D',
            '3DM_F43_1D', '3DM_F48_1D', '3DM_F54_2D',
        ],
        excluded_mice=[
            '3DM_D14', '3DM_D17', '3DM_F26', '3DM_F28', '3DM_F29',
            '3DM_F30', '3DM_F31', '3DM_F35', '3DM_F36', '3DM_F37',
            '3DM_F38', '3DM_F40', '3DM_F43', '3DM_F48', '3DM_F52',
            '3DM_F54', '3DM_J03', '3DM_J20', '3DM_J61',
        ],
    ),

    'RT': ExperimentConfig(
        experiment_id='RT',
        sessions=['1D', '2D', '3D'],
        matching_subdir='Footprint_matching',
        tables_subdir='INTENSE',
        delay_strategy=DELAY_STRATEGY,
    ),

    'BOF': ExperimentConfig(
        experiment_id='BOF',
        sessions=['1T', '2T', '3T', '4T', '5T'],
        nontrivial_matching=False,
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Group'],
    ),

    'FOF': ExperimentConfig(
        experiment_id='FOF',
        sessions=['1D', '2D', '3D'],
        nontrivial_matching=False,
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Line', 'Injection'],
    ),

    'MSS': ExperimentConfig(
        experiment_id='MSS',
        sessions=['2D_1T', '6D_1T'],
        nontrivial_matching=False,
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Line', 'Group', 'Injection'],
    ),

    'HOS': ExperimentConfig(
        experiment_id='HOS',
        sessions=['1D_1T', '2D_1T', '3D_1T', '4D_1T', '5D_1T'],
        nontrivial_matching=False,
        delay_strategy=DELAY_STRATEGY,
        mice_metadata=['Group'],
    ),
}
