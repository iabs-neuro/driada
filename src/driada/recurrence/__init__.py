"""Recurrence analysis for time series.

Provides Takens delay embedding, recurrence graph construction,
recurrence quantification analysis (RQA), and population-level
recurrence combination methods.
"""

from .embedding import takens_embedding, estimate_tau, estimate_embedding_dim
