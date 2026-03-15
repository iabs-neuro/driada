# DRIADA Notebook Validation Protocol

This protocol ensures all Colab/Jupyter notebooks in `notebooks/` are
correct, educational, and consistent with the DRIADA source code.

Adapted from `EXAMPLE_VALIDATION_PROTOCOL.md` for the notebook format.

## CRITICAL RULE: NO AUTO-VALIDATION

**YOU MUST GET USER APPROVAL BEFORE MARKING ANY NOTEBOOK AS VALIDATED**

- NEVER update NOTEBOOK_VALIDATION_STATUS.md without explicit user approval
- NEVER mark a notebook as DONE without user saying "validated" or "approved"
- ALWAYS present results and ASK: "Should I mark this as validated?"
- ALWAYS wait for user response before updating status files

---

## PHASE 0: GENERATOR EXECUTABILITY CHECK

**This phase catches broken generators early — before any manual review.**

Generators are the source of truth for notebooks. If a generator doesn't
run, the notebook is broken. This must pass before starting any per-notebook
validation.

### 0.1 Run All Generators
**Action**: Execute every generator script and verify exit code 0.

```bash
cd tools
for i in 01 02 03 04 05; do
  echo -n "create_notebook_${i}.py ... "
  python create_notebook_${i}.py && echo "OK" || echo "FAIL"
done
```

Any FAIL = stop and fix before proceeding.

### 0.2 Diff Generated vs Committed Notebooks
**Action**: Compare freshly generated notebooks against committed versions.

```bash
for i in 01 02 03 04 05; do
  python tools/create_notebook_${i}.py
  diff <(python -m json.tool notebooks/*_${i}_*.ipynb) <(python -m json.tool /tmp/generated_${i}.ipynb) || echo "DIVERGED: notebook ${i}"
done
```

If generators and committed notebooks diverge, the generator is the source
of truth — regenerate and commit.

**Deliverable**: All 5 generators execute cleanly; generated notebooks match committed versions.

---

## PHASE 1: PRE-EXECUTION ANALYSIS

### 1.1 API Currency Check
**Action**: Launch Explore agent to study relevant modules BEFORE checking
the notebook.
- Identify which DRIADA modules the notebook uses
- Study the current API for those modules
- Note any recent changes or deprecations

**Deliverable**: Understanding of current API patterns for comparison

### 1.2 Notebook Code Review
**Action**: Read the notebook file thoroughly
- Understand the workflow and learning objectives
- Identify all API calls, imports, attribute accesses, return-value unpackings
- Check that markdown cells accurately describe what the code cells do
- Note any comments or markdown claiming specific numerical results

**Deliverable**: Complete understanding of notebook structure

### 1.3 Generator Script Consistency
**Action**: If a generator script exists (`tools/create_notebook_XX.py`),
verify it matches the notebook.
- Run the generator and diff against the committed notebook
- If they diverge, the generator is the source of truth -- regenerate
- Any fix must be applied to BOTH the generator and the notebook

**Deliverable**: Confirmation that generator and notebook are in sync

### 1.4 Synthetic Data Analysis
**Action**: If the notebook generates synthetic data
- Study how data is created (functions, parameters)
- Understand expected properties (dimensionality, noise, selectivity)
- Verify generation is reproducible (seeds)
- Check that reduced parameters (shorter durations, fewer shuffles) are
  appropriate for a notebook demo

**Deliverable**: Understanding of data generation logic

---

## PHASE 2: FACT-CHECK AGAINST SOURCE CODE

This is the most critical phase for notebooks. Every API call must be verified
against the actual source code in `src/driada/`.

### 2.1 Import Verification
**Action**: For every import in the notebook
- Verify the function/class exists at the stated import path
- Check it is properly exported from `__init__.py` (or is a valid direct
  submodule import)
- Flag imports that work by accident (e.g. importing from a submodule
  that isn't in `__init__.py`)

### 2.2 Function Signature Verification
**Action**: For every function call
- Find the actual function definition in source code
- Verify every keyword argument name matches a real parameter
- Check for silently-absorbed `**kwargs` -- does the kwarg actually reach
  the code that uses it, or is it swallowed and ignored?
- Verify positional argument order

### 2.3 Return Value Verification
**Action**: For every return-value unpacking
- Find the actual return statement in source code
- Verify the number of values matches the unpacking
- Verify the order matches (e.g. `stats, significance, info, results`)
- For dict returns, verify every accessed key exists in the returned dict

### 2.4 Attribute Access Verification
**Action**: For every attribute access on returned objects
- Verify the attribute exists on the class
- Check the shape/type matches what the notebook expects
- Watch for attributes that are only set conditionally (e.g. after calling
  a specific method)

### 2.5 Write Fact-Check Report
**Action**: Write findings to `docs/plans/nbN_factcheck.md`

Format:
```markdown
# Notebook NN Fact-Check Report
## Summary
| Category | Checked | Issues Found |
## ISSUES FOUND (if any)
### BUG N: description
- Cell ID, line, code snippet
- What's wrong
- What the source code actually does
- Fix
## VERIFIED CORRECT (section-by-section)
## Final Verdict
```

**Deliverable**: Complete fact-check report with every API call verified

---

## PHASE 3: CODE QUALITY CHECK

### 3.1 Hardcoded Results Audit
**Action**: Search for hardcoded numerical results in markdown and comments
- FORBIDDEN: "We get MI = 2.5 bits" or "Dimension is 5"
- REQUIRED: Markdown describes WHAT is being done; code cells print ACTUAL results

```python
# BAD (in markdown cell):
# "Running INTENSE gives us 45 significant neurons..."

# GOOD (in markdown cell):
# "Run INTENSE and examine which neurons are significant."
# (then code cell prints the actual count)
```

**Deliverable**: List of hardcoded results requiring removal

### 3.2 Clean Educational Content
**Action**: Ensure markdown cells are clean educational material

FORBIDDEN content -- remove:
- Version history: "NEW:", "IMPROVED:", "UPDATED:", "FIXED:"
- API evolution: "unified API", "simplified API", "old method"
- Implementation details: "added caching", "refactored"
- Feature announcements: "showcasing", "demonstrating improvements"

REQUIRED content -- keep:
- Scientific concepts: "Compare representations", "Bootstrap testing"
- Method explanations: "This uses correlation distance"
- Parameter guidance: "Use more shuffles for real data"
- Workflow context: "Now that we have embeddings, we can test selectivity"

**Deliverable**: List of non-educational content removed

### 3.3 API Pattern Check
**Action**: Compare notebook API usage to current patterns (from Phase 1.1)
- Check for deprecated functions or parameter names
- Check for old-style initialization
- Verify imports use current module structure

**Deliverable**: List of API updates needed

### 3.4 Dead Code Detection
**Action**: Search for unused imports, unreachable code, and redundant cells
- Unused imports (imported but never called in any cell)
- Code cells that compute values never used in later cells
- Duplicate computations of the same thing

**Deliverable**: List of dead code removed

### 3.5 Duplicate Computation Detection
**Action**: Search for expensive operations repeated unnecessarily
- Same synthetic data generated multiple times
- Same embedding computed in separate cells
- Same significance test run redundantly

Notebooks may legitimately repeat patterns for pedagogical reasons (e.g.
showing the same workflow on different data). Only flag true waste.

**Deliverable**: List of duplicate operations removed

### 3.6 Duplicate Content Detection
**Action**: Scan for information stated in both markdown cells and code
comments/print statements.

Common patterns to catch:
- **Markdown explanation restated as code comment**: A markdown cell explains
  a concept, then the code cell below repeats it in a `#` comment
- **Banner comments echoing print statements**: `# === Step 3: Compute MI ===`
  followed by `print("3. Computing MI...")`
- **Parameter descriptions in both places**: Markdown describes what a
  parameter does, code comment on the same line says the same thing
- **Table data restated in print output**: A markdown table lists properties,
  then print statements repeat "Preserves: ..." for each item

Resolution rules:
- **Keep the explanation in markdown** (richer formatting, more visible)
- **Reduce code comments to structural markers** (e.g. section separators,
  variable grouping) not content restatements
- **Remove banner comments** when a print statement already announces the step
- **Inline code comments** may stay if they add context not in the markdown
  (e.g. "equivalent to exp.dynamic_features['speed']")

**Deliverable**: List of duplications resolved

### 3.7 Title case suppression
**Action**: Replace Title Case in markdown headers and print statements with
sentence case.
- BAD: `## Computing Feature-Feature Significance`
- GOOD: `## Computing feature-feature significance`

Allowed capitals: first word of heading, proper nouns (DRIADA, INTENSE),
class names when referring to code (TimeSeries, MVData), acronyms (MI, PCA,
RSA, DR, RDM).

**Deliverable**: List of title case violations fixed

### 3.8 Notebook structure check
**Action**: Verify notebook follows good Colab conventions
- First cell: pip install + imports (Colab install cell)
- Markdown cell before each code section explaining what comes next
- No empty cells or leftover output in committed notebook
- Cell execution order is linear (no forward references)
- All cells can run top-to-bottom without manual intervention

**Deliverable**: Structure assessment

### 3.9 Colab compatibility
**Action**: Verify notebook works in Colab environment
- No local file dependencies (all data is generated in-notebook)
- No `sys.path` manipulation
- Install cell uses `pip install driada` (or git+https for dev)
- `plt.show()` instead of `savefig()` for all plots
- Reasonable execution time (< 5 minutes total on Colab free tier)

**Deliverable**: Colab compatibility assessment

### 3.10 API documentation cross-links
**Action**: Verify that all DRIADA API references in markdown cells have
ReadTheDocs hyperlinks on first mention.

Every class, function, or method from `driada.*` that appears in a markdown
cell must be linked to its documentation page on first use:

```markdown
# GOOD (first mention is a hyperlink):
We build a [`RecurrenceGraph`](https://driada.readthedocs.io/en/latest/api/recurrence/recurrence_graph.html#driada.recurrence.recurrence_graph.RecurrenceGraph)
from the embedded time series.

# BAD (name appears in markdown without a link):
We build a `RecurrenceGraph` from the embedded time series.
```

Link format: `` [`ClassName`](https://driada.readthedocs.io/en/latest/api/{module}/{page}.html#{anchor}) ``

See the enrichment protocol (`docs/plans/2026-02-20-notebook-enrichment-protocol.md`)
for the full link map covering all DRIADA modules.

Rules:
- Link on **first mention only** -- subsequent uses can be plain backtick code
- Only link DRIADA API names, not standard library or third-party names
- Imports in code cells do not need links -- only markdown references

**Deliverable**: List of API references missing cross-links

### 3.11 User-friendly concept explanations
**Action**: Verify that every technical metric, algorithm, or domain term
is explained on first use in a way that helps a newcomer understand what
it means and why it matters.

REQUIRED -- explain on first mention:
- **Metrics and scores**: ARI, Jaccard, silhouette, modularity, MI,
  F1, p-value, etc. -- what does the number mean? What is a good value?
- **Algorithms**: Louvain, UMAP, PCA, FNN, TDMI, etc. -- one sentence
  on what it does, not just its name
- **Domain terms**: recurrence plot, Theiler window, copula, embedding
  dimension, etc. -- brief definition accessible to a non-specialist

```markdown
# GOOD:
Compare to ground truth with the **Adjusted Rand Index (ARI)** --
a measure of agreement between two clusterings, corrected for chance.
ARI = 1.0 is perfect match; ARI near 0 is no better than random.

# BAD:
Compare to ground truth with ARI.
```

Guidelines:
- Explain inline in the markdown cell where the concept first appears
- Keep explanations to 1-2 sentences -- enough to orient, not a textbook
- Include what a "good" or "bad" value looks like when applicable
- Wikipedia or textbook links are welcome for deeper reading
- Do not repeat the explanation on subsequent uses

**Deliverable**: List of unexplained terms found and explanations added

### 3.12 ASCII-only enforcement
**Action**: Search for and replace non-ASCII characters in code cells,
markdown cells, and print statements.
- PROBLEM: Unicode symbols cause UnicodeEncodeError on Windows consoles
  and render inconsistently across platforms
- REQUIRED: Use only ASCII characters in all code output and markdown

Common replacements:
```
Checkmarks:  U+2713 U+2714  ->  [OK] or >
Cross marks: U+2717 U+2718  ->  [FAIL] or X
Arrows:      U+2192 U+2190  ->  -> or <-
Bullets:     U+2022 U+00B7  ->  - or *
Box drawing: U+2502 U+2500  ->  | or -
Math:        U+2265 U+2264  ->  >= or <=
Em dash:     U+2014          ->  --
```

Detection:
```bash
python -c "
import json, sys
with open('notebooks/NN_name.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    for j, line in enumerate(cell['source']):
        for k, ch in enumerate(line):
            if ord(ch) > 127:
                print(f'Cell {i}, line {j}, col {k}: U+{ord(ch):04X} {ch!r}')
"
```

**Deliverable**: List of non-ASCII characters found and replaced

---

## PHASE 4: EXECUTION

### 4.1 Run Notebook
**Action**: Execute the notebook top-to-bottom

If running locally:
```bash
jupyter nbconvert --to notebook --execute notebooks/NN_name.ipynb
```

Or run cell-by-cell in Jupyter/Colab.

**Monitor**:
- Cell execution errors
- Warnings (especially DeprecationWarning)
- Execution time per section
- Memory usage for large operations

**Deliverable**: Execution log (success/failure, timing, warnings)

### 4.2 Error Handling
**Action**: If errors occur
- Capture full traceback
- Identify: import error, API mismatch, data shape issue, or logic error
- Check if the error matches a known issue from the fact-check (Phase 2)
- Fix in BOTH the notebook AND the generator script

**Deliverable**: Detailed error report if applicable

---

## PHASE 5: POST-EXECUTION VALIDATION

### 5.1 Visual Inspection
**Action**: Review ALL generated plots
- Overlapping elements (text, legends, labels)
- Visual consistency (colors, font sizes, style)
- Readability (axis labels visible, legend readable)
- Completeness (all expected panels present)
- Professional quality (no debug artifacts)

**Deliverable**: Visual quality report for each plot

### 5.2 Scientific Meaningfulness Check
**Action**: Think critically about whether results make scientific sense

What to check:

1. **Selectivity results**: Do the right neurons come out as significant?
   For synthetic data with ground truth, check detection rates.

2. **Dimensionality estimates**: Do they match expected intrinsic
   dimensionality? A circular manifold should be ~1-2, not 40.

3. **DR quality metrics**: Are preservation scores reasonable?
   Near-perfect (1.0) or near-zero suggest bugs.

4. **Network properties**: Does the functional network have expected
   density, clustering, community structure?

5. **Comparison baselines**: Are results compared to shuffled/null?
   Without a baseline, numbers are uninterpretable.

Critical questions:
- "Would a new researcher learn the right things from this?"
- "Are results compared to appropriate null hypothesis?"
- "Do visualization choices help understanding?"

**Deliverable**: Scientific meaningfulness assessment

---

## PHASE 6: REPORTING

### 6.1 Report Template

```markdown
## Notebook NN: [title]

**Status**: PASSED / ISSUES FOUND

**Fact-check**: N items verified, M issues found
**Execution**: Clean / warnings / errors
**Visual quality**: All plots clear / issues noted
**Scientific sense**: Results meaningful / concerns

**Issues fixed**:
1. ...

**Remaining issues**:
1. ...
```

### 6.2 User Validation (MANDATORY)

Present to user:
1. Notebook name and execution status
2. Fact-check summary (issues found and fixed)
3. Any remaining concerns
4. Generated plots (if relevant)

Wait for explicit user approval before updating any status files.

---

## CHECKLIST SUMMARY

For each notebook, verify:

**Phase 0 -- Generator executability**:
- [ ] All 5 generators run without errors (exit code 0)
- [ ] Generated notebooks match committed versions

**Phase 1 -- Pre-analysis**:
- [ ] Explored relevant API modules
- [ ] Read and understood notebook
- [ ] Generator script matches notebook (if applicable)
- [ ] Understood synthetic data generation

**Phase 2 -- Fact-check**:
- [ ] All imports resolve to valid paths
- [ ] All function signatures match (parameter names, counts)
- [ ] All return value unpackings match (count, order)
- [ ] All attribute accesses valid
- [ ] All dict key accesses valid
- [ ] Fact-check report written to `docs/plans/nbN_factcheck.md`

**Phase 3 -- Code quality**:
- [ ] No hardcoded results in markdown or comments
- [ ] Clean educational content (no version history, API evolution notes)
- [ ] Uses current API patterns
- [ ] No dead code (unused imports, unreachable cells)
- [ ] No unnecessary duplicate computations
- [ ] No duplicate content between markdown and code comments
- [ ] Sentence case in headers and print statements
- [ ] Good notebook structure (install cell, markdown before code, linear flow)
- [ ] Colab compatible (no local files, plt.show, reasonable runtime)
- [ ] API documentation cross-links on first mention of DRIADA classes/functions
- [ ] User-friendly explanations of metrics, algorithms, and domain terms
- [ ] ASCII-only characters in all code and markdown cells

**Phase 4 -- Execution**:
- [ ] Notebook executes without errors
- [ ] No unexpected warnings

**Phase 5 -- Post-execution**:
- [ ] All plots visually inspected
- [ ] No overlapping elements or visual issues
- [ ] Results make scientific sense
- [ ] Appropriate baselines present

**Phase 6 -- Reporting**:
- [ ] Presented results to user
- [ ] User explicitly approved
- [ ] Status file updated ONLY after approval

---

## EXECUTION TEMPLATE

```
0. Run all 5 generators — verify exit code 0, diff against committed notebooks
1. Explore relevant DRIADA modules for the notebook's topic
2. Read the notebook
3. Fact-check every API call against source code (Phase 2)
4. Write fact-check report to docs/plans/nbN_factcheck.md
5. Check code quality (dead code, title case, educational content)
6. Execute notebook top-to-bottom
7. Inspect all generated plots
8. Check scientific meaningfulness
9. Fix issues in BOTH notebook and generator script
10. Present report to user
11. Wait for user approval
12. Update status file only after approval
```
