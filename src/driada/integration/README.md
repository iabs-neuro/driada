# Integration Module

## Overview

The integration module bridges single-neuron analysis (INTENSE) with population-level dimensionality reduction, revealing how individual neuron properties contribute to population dynamics.

## Key Functions

### get_functional_organization()
Analyzes how neurons contribute to embedding components:
- Maps neuron selectivity to principal components
- Identifies functional clusters
- Quantifies component specialization

### compare_embeddings()
Compares functional organization across different DR methods:
- Cross-method consistency analysis
- Identifies method-specific features
- Helps choose optimal DR approach

## Example Usage

```python
from driada.integration import get_functional_organization
from driada.intense import compute_embedding_selectivity

# First, analyze how neurons respond to embedding components
emb_results = compute_embedding_selectivity(
    exp,
    embedding_methods=['pca', 'umap'],
    n_shuffles=1000,
    ds=5
)

# Then analyze functional organization
pca_org = get_functional_organization(
    exp,
    'pca',
    intense_results=emb_results['pca']['intense_results']
)

print(f"Component importance: {pca_org['component_importance']}")
print(f"Neurons participating: {pca_org['n_participating_neurons']}")
print(f"Mean components per neuron: {pca_org['mean_components_per_neuron']}")

# Compare across methods
from driada.integration import compare_embeddings

comparison = compare_embeddings(
    exp,
    ['pca', 'umap'],
    intense_results_dict={
        'pca': emb_results['pca']['intense_results'],
        'umap': emb_results['umap']['intense_results']
    }
)
```

## Analysis Pipeline

1. **Run INTENSE on features** - Identify what neurons encode
2. **Apply DR** - Extract population structure  
3. **Run INTENSE on components** - How neurons contribute to structure
4. **Analyze organization** - Link single cells to population

## Key Insights

This module helps answer:
- Which neurons drive each principal component?
- Are functional groups preserved across DR methods?
- How does single-neuron selectivity relate to population structure?
- Which DR method best preserves functional organization?