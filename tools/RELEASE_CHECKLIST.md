# DRIADA Release Checklist

> **Purpose:** Comprehensive checklist to run before every major release.
> Every item traces to a real bug found during v1.0 preparation
> (commits `20bc910..aefcc22`, ~90 commits of fixes).

---

## 1. Version & metadata consistency

- [ ] Version matches in ALL locations:
  - `src/driada/__init__.py` (`__version__`)
  - `pyproject.toml` (`version`)
  - `docs/conf.py` (`release`)
  - `CITATION.cff` (`version`)
  - `CHANGELOG.md` (latest entry header)
- [ ] `CHANGELOG.md` has actual date (not placeholder)
- [ ] `CITATION.cff` has actual `date-released` (not placeholder)
- [ ] `CITATION.cff` ORCID is real (not `0000-0000-0000-0000`)
- [ ] Publication dates match release year (if applicable)
- [ ] README BibTeX citation key and year are current
- [ ] README has no "pre-release" or stale version language
- [ ] No hardcoded counts that go stale (example counts, notebook counts,
  test counts) — use vague language or omit
- [ ] Python version requirement is correct everywhere (README, pyproject.toml, classifiers)
- [ ] Project name/title consistent across all surfaces:
  - `CITATION.cff` title
  - README acronym expansion (line 3)
  - `docs/index.rst` description
  - BibTeX blocks in README, `docs/license.rst`, `PUBLICATIONS.md`
  - `src/driada/__init__.py` module docstring
- [ ] New modules referenced in ALL key documentation:
  - README Key Capabilities list
  - `docs/index.rst` Key features list
  - `docs/examples.rst` (with all example scripts)
  - `docs/quickstart.rst` Next steps
  - Notebook 00 "Next steps" list
  - Class docstrings (TimeSeries methods, Network subclasses in See Also)

**Found:** 4 files had different version numbers (0.6.6, 0.7.2, 1.0.0 simultaneously),
placeholder dates in CITATION.cff, stale BibTeX key, README still saying "pre-release."
In v1.1.0: CITATION.cff had different title than bibtex everywhere; recurrence missing
from README Key Capabilities, index.rst Key features, quickstart Next steps, TimeSeries
docstring, Network See Also.

---

## 2. CI: full green on all workflows

Run these commands locally and verify they match GitHub CI:

```bash
# 1. Tests (the main gate)
python -m pytest tests/ -q

# 2. Documentation consistency
python tools/doc_consistency_checker.py src/driada --format text

# 3. Documentation references
python tools/verify_doc_references.py

# 4. Documentation examples analysis
python tools/analyze_doc_examples.py

# 5. Doctests (per-module)
python tools/run_doctests.py quickstart --timeout 60
python tools/run_doctests.py api/intense --timeout 60
# ... (all modules from run-doctests.yml matrix)
```

- [ ] `pytest`: 0 failures, 0 collection errors
- [ ] `doc_consistency_checker`: 0 critical issues (warnings OK)
- [ ] `verify_doc_references`: all references valid
- [ ] `analyze_doc_examples`: 0 issues
- [ ] `run_doctests`: no encoding errors or import failures
- [ ] GitHub Actions: all workflows green on latest commit

**Found:** test collection errors from module-level fixtures, RST encoding errors
from non-ASCII characters, flaky cross-platform assertions, division-by-zero
in timing tests, NumPy scalar repr changes breaking doctests.

---

## 3. Test health

- [ ] No module-level code in test files that can fail at import time
  (warmup calls, fixture generation at module scope)
- [ ] All test fixtures use realistic parameter ranges
  (duration >= 30s at fps=10 for shuffle mask compatibility:
   `n_frames > 2 * CA_SHIFT_N_TOFF * default_t_off * fps`)
- [ ] Performance tests guard against zero-time divisions
  (`max(elapsed, 1e-9)` before computing speedup ratios)
- [ ] Cross-platform assertions use tolerant thresholds
  (graph spectral tests, embedding quality tests are
   non-deterministic across OS/CPU)
- [ ] Stochastic tests have sufficient tolerance
  (random generation can produce wide variance at small sample sizes)
- [ ] No flaky tests on either Ubuntu or Windows runners
- [ ] Run `tools/validate_tests.py` -- 0 RED findings
- [ ] Synthetic generators produce detectable selectivity at committed seeds:
  ```bash
  python -c "
  from driada import generate_synthetic_exp
  from driada.intense import compute_cell_feat_significance
  exp = generate_synthetic_exp(n_dfeats=2, n_cfeats=2, nneurons=10, duration=60, seed=42)
  stats, significance, info, intense_res = compute_cell_feat_significance(
      exp, n_shuffles_stage1=200, n_shuffles_stage2=200)
  n_sig = sum(1 for cell in significance.values() for sig in cell.values() if sig)
  assert n_sig > 0, f'No significant pairs found (seed=42)'
  print(f'OK: {n_sig} significant pairs')
  "
  ```

**Found:** warmup fixtures with duration too short for exclusion zones,
timing test divided by zero on fast CI, spectral threshold failed
on one OS but passed on another, stochastic active fraction test
too tight for random variance, 36 RED test quality findings
(zero-assertion tests, loose tolerances for exact results, dead code).
Synthetic generators produced 17-50% undetectable neurons depending
on feature type and tuning parameters.

---

## 4. Encoding & character safety

- [ ] No bare `0x97` bytes (Windows-1252 em dash) in `.rst` or `.py` files:
  ```bash
  python -c "
  import glob
  for p in ['docs/**/*.rst', 'src/**/*.py']:
      for f in glob.glob(p, recursive=True):
          with open(f, 'rb') as fh:
              data = fh.read()
              pos = 0
              while True:
                  idx = data.find(b'\x97', pos)
                  if idx == -1: break
                  if idx > 0 and data[idx-1] == 0xc3: pos = idx+1; continue  # UTF-8 multiply sign
                  print(f'FAIL: {f} pos {idx}'); break
                  pos = idx + 1
  print('OK')
  "
  ```
- [ ] No em dashes in notebook markdown cells (use `--` instead)
- [ ] All `.py` and `.rst` files are valid UTF-8
- [ ] Tools use `encoding='utf-8'` for file I/O (not system default codepage)
- [ ] No Unicode emoji in tool output (use ASCII equivalents like `[OK]`, `[FAIL]`)

**Found:** em dash bytes caused UnicodeDecodeError on GitHub CI,
notebook em dashes caught by manual validation, all 4 doc tools
crashed on Windows cp1251 codepage due to emoji in print().

---

## 5. Notebook validation (6-phase protocol)

Run the full protocol from `tools/NOTEBOOK_VALIDATION_PROTOCOL.md` on all notebooks.

### Phase 0: Automated execution check
- [ ] All notebooks execute top-to-bottom without errors:
  ```bash
  python tools/verify_notebooks.py -v
  ```
  All 7 must report PASS. Fix any failures before proceeding to manual review.

### Phase 1: Pre-analysis
- [ ] All imports resolve against current installed package
- [ ] All function signatures match current source code
- [ ] No `**kwargs` silently swallowing wrong parameter names

### Phase 2: Fact-check
- [ ] Every function call uses correct parameter names
- [ ] Every return value access uses correct dict keys
- [ ] Every attribute access exists on the object
- [ ] API doc links point to pages that exist
- [ ] Comments match what the code actually does
- [ ] No development-log text in markdown ("NEW:", "IMPROVED:", "FIXED:")

### Phase 3: Code quality
- [ ] No dead variables (assigned but never read)
- [ ] No unused imports
- [ ] Headings in sentence case (not Title Case)
- [ ] No hardcoded results that could go stale
- [ ] No programmatic `.title()` calls on dynamic text

### Phase 4: Execution
- [ ] Each notebook executes top-to-bottom without errors
- [ ] No cell takes > 120s (or is documented as expected-slow)
- [ ] No warnings that indicate real problems
- [ ] `MultiTimeSeries` features are not passed to `TimeSeries`-only functions

### Phase 5: Post-execution
- [ ] Plots render and look scientifically reasonable
- [ ] Printed numerical results are in expected ranges

### Phase 6: Structure & Colab compatibility
- [ ] First cell is `pip install` + imports
- [ ] Markdown cell before each code section
- [ ] No empty cells or leftover output
- [ ] No local file dependencies (`sys.path` manipulation, local data files)
- [ ] Uses `plt.show()` not `savefig()` for inline display
- [ ] Total execution time < 5 minutes on Colab free tier

### Phase 7: Sign-off
- [ ] User has reviewed and approved each notebook

**Found:** random noise calcium data caused 617s timeout, MultiTimeSeries crash
in plotting, factually wrong graph algorithm label, Title Case in all headers,
dead variables and wrong dict key access, 4 unused imports, development-log
markers in educational content.

---

## 6. Generator-notebook sync

- [ ] Run each generator and diff output against checked-in notebook:
  ```bash
  for i in 00 01 02 03 04 05 06; do
    cp notebooks/${i}_*.ipynb /tmp/nb${i}_before.ipynb
    python tools/create_notebook_${i}.py
    python -c "
  import json
  with open('/tmp/nb${i}_before.ipynb') as f: old = json.load(f)
  with open([x for x in __import__('glob').glob('notebooks/${i}_*.ipynb')][0]) as f: new = json.load(f)
  old_src = [''.join(c['source']) for c in old['cells']]
  new_src = [''.join(c['source']) for c in new['cells']]
  diffs = [i for i,(o,n) in enumerate(zip(old_src,new_src)) if o!=n]
  if len(old_src)!=len(new_src): print(f'NB${i}: CELL COUNT {len(old_src)} vs {len(new_src)}')
  elif diffs: print(f'NB${i}: {len(diffs)} cell diffs')
  else: print(f'NB${i}: OK')
  "
  done
  ```
- [ ] All generators report "OK" (zero source cell diffs)
- [ ] Cross-notebook links use Colab URLs (not local `.ipynb` paths)
- [ ] All Colab URLs point to `blob/main/` (not `blob/dev/`)
- [ ] All `!pip install` lines use `@main` (not `@dev`)
- [ ] Any direct notebook edit is reflected back into the generator

**Found:** all generators drifted out of sync during documentation updates.
Direct notebook edits were not propagated to generators.
Local `.ipynb` cross-links don't work in Colab.
During dev, Colab URLs and pip install may temporarily point to `dev` branch --
revert all to `main` before release.

---

## 7. Documentation build

- [ ] `sphinx-build` completes without errors:
  ```bash
  cd docs && make html 2>&1 | grep -E "ERROR|WARNING" | head -20
  ```
- [ ] No broken `:doc:`, `:func:`, `:class:`, `:meth:` references
- [ ] No undefined labels or missing targets
- [ ] RST code examples are syntactically valid Python
  (no references to removed functions, no undefined variables)
- [ ] `myst_parser` is in docs build dependencies (if using `.md` files)
- [ ] Doctest expected outputs match current NumPy scalar representation
  (e.g., `np.float64(1.0)` not bare `1.0`)

**Found:** quickstart.rst had undefined variables and references to removed
functions. Cross-references to nonexistent doc targets. NumPy repr
changes broke doctests silently.

---

## 8. Package & distribution

- [ ] `python -m build` succeeds:
  ```bash
  pip install build && python -m build
  ```
- [ ] `twine check dist/*` passes (no warnings)
- [ ] No stale `dist/` artifacts from previous builds
- [ ] No `NUL` files (Windows artifacts) in `examples/`
- [ ] No credentials, `.env`, or large data files staged
- [ ] `.gitignore` covers `dist/`, `*.egg-info/`, `coverage.xml`, etc.
- [ ] `pip install -e ".[dev]"` works cleanly
- [ ] `[dev]` extra includes all test/doc dependencies
- [ ] pytest config is in `pyproject.toml` (not a separate `pytest.ini`)
- [ ] `publish.yml` workflow exists and triggers on tags/releases

**Found:** stale `dist/` artifacts and Windows `NUL` files in repo.
`[dev]` extra was missing. pytest config was split across files.
No publish workflow existed initially.

---

## 9. Source code hygiene

- [ ] No `**kwargs` functions silently swallowing wrong parameter names
  (search for `**kwargs` in public API and verify they either
   forward to a known target or raise on unknown keys)
- [ ] All public functions have docstrings matching their signatures
  (run `doc_consistency_checker` with 0 critical issues)
- [ ] No private attributes accessed in notebooks
  (e.g., `._reconstructed` -- use public API instead)
- [ ] All public utilities are exported from `__init__.py`
- [ ] README code examples actually run:
  ```bash
  python -c "from driada.network import Network; print('OK')"
  python -c "from driada.dim_reduction import ProximityGraph; print('OK')"
  python -c "from driada.intense import compute_cell_cell_significance; print('OK')"
  ```
- [ ] Test assertions match current API:
  - Attribute names (e.g., `exp.n_cells` not `exp.n`)
  - Error message text
  - Feature/parameter names after refactors
  - JSON round-trip key types (int keys become strings)
- [ ] Planned deprecations executed or still warned
  (check `tools/FUTURE.md` deprecation schedule against current code)

**Found:** t-SNE `perplexity` and UMAP `random_state` silently ignored via
`**kwargs`. 28 critical docstring-signature mismatches. README had 3 bugs.
`visualize_circular_manifold` not exported. Test assertions referenced
removed sigmoid parameters, wrong attribute names, stale error messages.
Planned deprecations (`joint_distr`) not executed on schedule.

---

## 10. Examples cleanup

Run the full protocol from `tools/EXAMPLE_VALIDATION_PROTOCOL.md` on all example scripts.

- [ ] No example scripts with broken API calls
- [ ] No redundant/duplicate example scripts
- [ ] No non-ASCII characters in example scripts
- [ ] All example output files are `.gitignore`d or tracked intentionally
- [ ] Module READMEs list current exports (not stale lists)
- [ ] All outputs stay inside the example's own folder (no writes to parent/root)
- [ ] No hardcoded or absolute file paths (use `pathlib` / `os.path`)
- [ ] No untested code shown in comments or docstrings
- [ ] No duplicate expensive computations
- [ ] Each script has a clear `__main__` entry point

**Found:** example script had 4 broken API calls (`net.g` instead of
`net.graph`). Redundant scripts duplicated functionality. Non-ASCII
characters in example output. Module READMEs had missing exports.
Output files written outside example folders. Hardcoded paths.

---

## 11. Test validation (full suite)

Run the complete test suite and validate test quality before release:

```bash
# 1. Full pytest run
python -m pytest tests/ -q --tb=short

# 2. Test quality analysis
python tools/validate_tests.py
```

- [ ] `pytest tests/` passes with 0 failures, 0 errors, 0 collection errors
- [ ] `validate_tests.py` reports 0 RED findings
- [ ] No tests marked `@pytest.mark.skip` without a tracked issue
- [ ] Synthetic generator round-trip produces detectable selectivity (section 3 command)
- [ ] Slow/integration tests pass when run explicitly:
  ```bash
  python -m pytest tests/ -m "slow or integration" -q
  ```

**Why a separate step:** Sections 2-3 cover CI and test health rules. This step
is the actual gate — run everything locally, confirm the numbers, sign off.

---

## 12. Final pre-tag steps

- [ ] All sections 1-11 are checked off
- [ ] `git status` is clean (no uncommitted changes)
- [ ] `git log origin/main..HEAD` is empty (everything pushed)
- [ ] GitHub Actions all green on the latest commit
- [ ] CHANGELOG entry is finalized with actual date
- [ ] Create git tag: `git tag -a v{VERSION} -m "Release v{VERSION}"`
- [ ] Push tag: `git push origin v{VERSION}`
- [ ] Create GitHub Release (triggers `publish.yml` -> PyPI)
- [ ] Verify package appears on PyPI: `pip install driada=={VERSION}`
- [ ] Verify ReadTheDocs builds successfully

---

## Automation opportunities

Issues that could be caught by CI but currently are not:

| Gap | Proposed CI step |
|-----|-----------------|
| Notebook execution failures | `python tools/verify_notebooks.py` (manual); promote to CI with timeout |
| Generator-notebook drift | Regenerate + diff in CI, fail on mismatch |
| Non-ASCII in docs/source | Pre-commit hook scanning for bare `0x97` and em dashes |
| Title case violations | Custom linter for notebook markdown headers |
| `**kwargs` parameter forwarding | AST-based check that `**kwargs` are forwarded or validated |
| Stale docstrings | Promote `doc_consistency_checker` critical issues to CI failure |
| Test quality regression | Run `validate_tests.py` in CI, fail on RED findings |
| NumPy doctest compat | Version-aware expected output or `+NORMALIZE_WHITESPACE` |
