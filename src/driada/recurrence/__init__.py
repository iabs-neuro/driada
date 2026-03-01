"""Recurrence analysis for time series.

Provides Takens delay embedding, recurrence graph construction,
recurrence quantification analysis (RQA), and population-level
recurrence combination methods.
"""

from .embedding import takens_embedding, estimate_tau, estimate_embedding_dim
from .recurrence_graph import RecurrenceGraph
from .visibility import VisibilityGraph
from .opn import OrdinalPartitionNetwork
from .rqa import compute_rqa
from .population import population_recurrence_graph, pairwise_jaccard_sparse
from .plotting import plot_recurrence
