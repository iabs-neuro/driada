# DRIADA Example Validation Protocol

This protocol ensures all examples are production-ready, user-friendly, and follow consistent standards.

## ⚠️ CRITICAL RULE: NO AUTO-VALIDATION ⚠️

**YOU MUST GET USER APPROVAL BEFORE MARKING ANY EXAMPLE AS VALIDATED**

- ❌ **NEVER update EXAMPLE_VALIDATION_STATUS.md without explicit user approval**
- ❌ **NEVER mark example as [x] DONE without user saying "validated" or "approved"**
- ❌ **NEVER move to next example without user permission**
- ✅ **ALWAYS present results and ASK: "Should I mark this as validated?"**
- ✅ **ALWAYS wait for user response before updating status files**

**After running example**: Present results → Wait for user approval → ONLY THEN update status

---

## PHASE 1: PRE-EXECUTION ANALYSIS

### 1.1 API Currency Check
**Action**: Launch Explore agent to study relevant modules BEFORE checking the example
- Identify which DRIADA modules the example uses (Experiment, INTENSE, dimensionality reduction, etc.)
- Study the current API for those modules
- Note any recent changes or deprecations

**Deliverable**: Understanding of current API patterns for comparison

### 1.2 Example Code Review
**Action**: Read the example file thoroughly
- Understand the workflow and objectives
- Identify all output operations (plots, data saves, reports)
- Check imports and API usage patterns
- Note any comments claiming specific results

**Deliverable**: Complete understanding of example structure

### 1.3 Synthetic Data Analysis (if applicable)
**Action**: If example uses synthetic data generation
- Study how data is created (functions, parameters)
- Understand expected properties (dimensionality, noise level, selectivity patterns)
- Verify generation is reproducible (seeds, deterministic)

**Deliverable**: Understanding of data generation logic

---

## PHASE 2: FILE ORGANIZATION CHECK

### 2.1 Output Path Analysis
**Action**: Verify all file output paths in code
- **REQUIREMENT**: All outputs MUST go inside the example's own folder
- **ALLOWED**: One level of subfolders (e.g., `examples/my_example/plots/`)
- **FORBIDDEN**:
  - Outputs to root directory
  - Outputs to parent directory
  - Outputs to other examples' folders
  - Recursive nested subfolders (more than one level deep)

**Check**:
```python
# GOOD:
output_dir = "examples/my_example/"
output_dir = "examples/my_example/plots/"

# BAD:
output_dir = "."  # root
output_dir = "../"  # parent
output_dir = "examples/my_example/results/plots/data/"  # too deep
```

**Deliverable**: Confirmation all paths comply, or list of violations

### 2.2 Hardcoded Path Detection
**Action**: Check for absolute paths or machine-specific paths
- Should use relative paths from example location
- Should work cross-platform (use `pathlib` or `os.path`)

**Deliverable**: List any hardcoded paths requiring fixes

---

## PHASE 3: CODE QUALITY CHECK

### 3.1 Hardcoded Results Audit
**Action**: Search for hardcoded results in text/comments
- **FORBIDDEN**: Comments like "# We get 95% accuracy" or "# Dimension is 5"
- **REQUIRED**: Explanatory comments about WHAT is being done
- **REQUIRED**: Print statements to show ACTUAL results

**Examples**:
```python
# BAD:
# Calculate mutual information (we get MI=2.5 bits)
mi = calculate_mi(x, y)

# GOOD:
# Calculate mutual information between variables
mi = calculate_mi(x, y)
print(f"Mutual Information: {mi:.3f} bits")
```

**Deliverable**: List of hardcoded results requiring removal

### 3.2 User-Friendliness Check
**Action**: Evaluate entry point and usability
- Is the main execution clear? (e.g., `if __name__ == "__main__":`)
- Are parameters documented?
- Is the workflow easy to follow?
- Are there helpful print statements showing progress?

**Deliverable**: User-friendliness assessment and improvement suggestions

### 3.3 API Pattern Check
**Action**: Compare example API usage to current patterns (from Phase 1.1)
- Check for deprecated functions
- Check for outdated parameter names
- Check for old-style initialization
- Verify imports use current module structure

**Deliverable**: List of API updates needed

### 3.4 ASCII-Only Enforcement (CRITICAL)
**Action**: Search for and replace non-ASCII characters in print statements and strings
- **PROBLEM**: Fancy Unicode symbols (✓, ✗, →, •, etc.) cause UnicodeEncodeError on Windows consoles
- **REQUIRED**: Use only ASCII characters in all output
- **MUST FIX BEFORE EXECUTION**: This prevents runtime errors

**Common replacements**:
```python
# BAD:
print("✓ Success")
print("✗ Failed")
print("→ Next step")
print("• Item")

# GOOD:
print("[OK] Success")
print("[FAIL] Failed")
print("-> Next step")
print("- Item")
```

**Search patterns** to check:
- Checkmarks: `✓` `✔` → Replace with `[OK]` or `>`
- Cross marks: `✗` `✘` → Replace with `[FAIL]` or `X`
- Arrows: `→` `←` `↔` → Replace with `->` `<-` `<->`
- Bullets: `•` `·` → Replace with `-` or `*`
- Box drawing: `│` `─` `┌` etc. → Replace with `|` `-` `+`
- Math symbols: `≥` `≤` `≠` → Replace with `>=` `<=` `!=`

**Action steps**:
1. Grep for common Unicode characters in the example file
2. Replace ALL non-ASCII symbols with ASCII alternatives
3. Test that file contains only ASCII (use `file.encode('ascii')` check)

**Deliverable**: List of replacements made (if any)

### 3.5 Clean Educational Comments (CRITICAL)
**Purpose**: Examples are educational material for users, not development logs

**FORBIDDEN content** - Remove from examples:
- Version history markers: "NEW:", "IMPROVED:", "UPDATED:", "FIXED:", "OLD:"
- API evolution mentions: "unified API", "simplified API", "old API", "new method"
- Implementation details: "added caching", "speedup applied", "changed seed", "refactored"
- Performance logs: timing comparisons, speedup factors (unless RSA/algorithm concept)
- Feature announcements: "showcasing", "demonstrating improvements", "recent changes"
- Change summaries: "Summary of Improvements" sections at the end

**REQUIRED content** - Keep clean educational focus:
- Scientific concepts: "Compare representations", "Bootstrap testing", "Trial structure"
- Method explanations: "This uses correlation distance", "Dendrogram shows clustering"
- Parameter meanings: "duration=300  # 5 minutes", "n_bootstrap=100  # Use more for real analysis"
- Workflow steps: "Generate data", "Compute RDM", "Visualize results"

**Examples of violations**:
```python
# BAD - Implementation logs:
"""
This example demonstrates:
1. NEW: Unified API with automatic data type detection
2. NEW: Caching support for repeated computations
3. IMPROVED: Simplified rsa_compare() function
"""

print("1. NEW: Simplified rsa_compare() API for common use case")
print("2. Unified API (compute_rdm_unified) - automatic data type detection")

# NEW: Use unified API that automatically detects Experiment object
rdm = rsa.compute_rdm_unified(exp, items="stimulus")

# NEW: Demonstrate caching
print("First computation: 0.045s")
print("Cached computation: 0.001s")
print("Speedup: 45x")

# GOOD - Clean educational content:
"""
This example demonstrates:
1. Computing RDMs from neural data
2. Comparing representations between conditions
3. Statistical testing with bootstrap methods
"""

print("1. Computing RDMs from neural data")
print("2. Comparing representations between conditions")

# Compute RDM from Experiment object
rdm = rsa.compute_rdm_unified(exp, items="stimulus")

# Compare with statistical significance
print("Running bootstrap test...")
```

**Action steps**:
1. Grep for forbidden patterns: `grep -i "NEW:|IMPROVED:|unified|simplified|showcasing|speedup" example.py`
2. Review docstrings - remove API history, keep scientific concepts
3. Review print statements - remove change announcements, keep results
4. Review comments - remove implementation details, keep explanations
5. Remove "Summary of Improvements" or similar sections
6. Remove timing/performance comparisons unless they demonstrate scientific concepts

**Deliverable**: List of implementation logs removed

### 3.6 Dead Code Detection (CRITICAL)
**Action**: Search for unused functions, imports, and unreachable code BEFORE execution
- **PROBLEM**: Examples accumulate dead code from refactoring, making them confusing and misleading
- **REQUIRED**: Remove ALL dead code before validation

**What to check**:
1. **Unused imports**: Imports from libraries never used in code (e.g., sklearn imports when using MVData API)
2. **Unused functions**: Functions defined but never called
3. **Commented-out code**: Old code that should be deleted
4. **Unreachable code**: Code after return statements or in never-true conditionals

**Detection strategy**:
```python
# Search for function definitions
grep "^def " example.py

# For each function, search for calls
grep "function_name(" example.py

# If only 1 match (the definition), it's unused → DELETE
```

**Common dead code patterns**:
- Legacy sklearn wrapper functions when MVData API used
- Old visualization functions replaced by plot_embedding_comparison
- Manual implementations replaced by library functions
- Parameter processing code for removed options

**Action steps**:
1. List all function definitions in example
2. Search for calls to each function
3. Mark functions with only 1 match (definition) as DEAD
4. Verify by reading context - ensure truly unused
5. DELETE dead functions and their imports

**Deliverable**: List of dead code removed

### 3.7 Untested Code in Comments/Docstrings (CRITICAL)
**Action**: Search for code examples in comments/docstrings that are not actually executed in the script
- **PROBLEM**: Examples show API usage but don't validate it works, creating maintenance burden
- **ANTI-PATTERN**: Printing example code without running it
- **REQUIRED**: All code shown to users must be validated by execution

**What to check**:
1. **Print statements with code examples**: `print("   emb = mvdata.get_embedding(...)")`
2. **Docstring code examples**: Code in triple-quoted strings not tested
3. **Commented example code**: `# Example: result = function(param)`

**Allowed exceptions**:
- Code snippets that ARE executed elsewhere in the same script
- Syntax examples showing parameter names (e.g., "method='pca', dim=3")
- Error handling examples (intentionally showing wrong usage)

**Detection strategy**:
```python
# Search for print statements with code
grep 'print.*=.*(' example.py

# Check if the printed code is actually executed
# Look for corresponding execution without print
```

**Examples of violations**:

**BAD - Untested code in comments:**
```python
def show_advanced_usage():
    print("Advanced example:")
    print("  emb = mvdata.get_embedding(method='pca', dim=50)")
    print("  # ... more code")
    # ❌ This code is never run - might be broken!
```

**GOOD - Code is executed and validated:**
```python
def show_advanced_usage():
    print("Advanced example:")
    print("  emb = mvdata.get_embedding(method='pca', dim=50)")
    emb = mvdata.get_embedding(method='pca', dim=50)  # ✓ Actually runs it
    print(f"  -> Result shape: {emb.coords.shape}")  # ✓ Shows it worked
```

**Action steps**:
1. Grep for `print` statements containing code examples
2. For each example, verify it's executed somewhere in the script
3. If not executed → either delete the print or add execution + validation
4. Ensure all "advanced usage" sections actually run their examples

**Deliverable**: List of untested code either removed or converted to tested code

### 3.8 Duplicate/Redundant Computation Detection (CRITICAL)
**Action**: Search for expensive operations that are repeated unnecessarily
- **PROBLEM**: Code generates/computes the same data multiple times, wasting resources and creating confusion
- **ANTI-PATTERN**: Calling expensive functions (data generation, model training, etc.) multiple times with same/similar parameters
- **REQUIRED**: Expensive operations should be done ONCE and reused

**What to check**:
1. **Duplicate data generation**: Same synthetic data function called multiple times
2. **Redundant computations**: Same calculation performed in multiple places
3. **Unused results**: Function called but result never used (suggests duplication elsewhere)
4. **Data reloading**: Same file loaded multiple times instead of reusing variable

**Detection strategy**:
```python
# Search for expensive function calls
grep "generate.*data\|generate.*exp\|load.*data" example.py

# Check if called multiple times
# If same function appears >1 time with similar params → likely duplicate
```

**Common duplicate patterns**:
- Data generated in helper function, then again in main()
- Training/fitting same model multiple times
- Computing same embedding/reduction multiple times
- Loading same dataset in different parts of code

**Examples of violations**:

**BAD - Duplicate data generation:**
```python
def visualize_results():
    # Generate data for visualization
    data = generate_synthetic_data(n=1000, seed=42)
    plot_data(data)

def main():
    # Generate SAME data again
    data = generate_synthetic_data(n=1000, seed=42)
    results = analyze_data(data)
    visualize_results()  # ❌ Generates data AGAIN inside
```

**GOOD - Generate once, reuse:**
```python
def visualize_results(data):
    plot_data(data)

def main():
    # Generate data ONCE
    data = generate_synthetic_data(n=1000, seed=42)
    results = analyze_data(data)
    visualize_results(data)  # ✓ Reuses data
```

**BAD - Function called but result unused:**
```python
def analyze_and_plot(data):
    # Generates embeddings internally
    embedding = compute_embedding(data)
    plot_embedding(embedding)
    return None  # ❌ Doesn't return embedding

def main():
    data = load_data()
    analyze_and_plot(data)  # Computes embedding
    # Need embedding for metrics
    embedding = compute_embedding(data)  # ❌ Computes AGAIN
    metrics = evaluate(embedding)
```

**GOOD - Compute once, return and reuse:**
```python
def analyze_and_plot(data):
    embedding = compute_embedding(data)
    plot_embedding(embedding)
    return embedding  # ✓ Return for reuse

def main():
    data = load_data()
    embedding = analyze_and_plot(data)  # ✓ Compute once
    metrics = evaluate(embedding)  # ✓ Reuse result
```

**Action steps**:
1. Grep for expensive operation patterns (generate_, compute_, train_, fit_, load_)
2. For each pattern, count occurrences and check parameters
3. If same function called multiple times with similar params:
   a. Check if results are reused (variable passed between calls)
   b. If NOT reused → mark as DUPLICATE
4. For each duplicate:
   a. Determine which call should be kept (usually in main())
   b. Refactor to pass result to functions that need it
   c. Remove redundant calls
5. Verify functions return computed results for reuse (not just None)

**Detection checklist**:
- [ ] All `generate_*` functions called only once per unique parameter set
- [ ] All expensive computations (embeddings, models) computed once and passed around
- [ ] Functions that compute expensive results return those results (not None)
- [ ] No data loading from same file multiple times
- [ ] No redundant calculations that could be avoided with variables

**Deliverable**: List of duplicate operations removed with before/after code organization

### 3.9 Title Case Suppression (CRITICAL)
**Action**: Replace Title Case Capitalisation In Headers And Print Statements with sentence case
- **PROBLEM**: LLM-generated text defaults to capitalising every word ("Pairwise Mutual Information", "Creating Visualizations"). Humans don't write like that and don't like reading it.
- **REQUIRED**: Use sentence case everywhere — capitalise the first word only (plus proper nouns and acronyms)

**What to check**:
1. **Print headers**: `print("Pairwise Mutual Information")` -> `print("Pairwise mutual information")`
2. **Section titles in docstrings**: `Sections:\n1. Create TimeSeries From Numpy Arrays` -> `1. Create TimeSeries from numpy arrays`
3. **Figure titles**: `ax.set_title("Degree Distribution")` -> `ax.set_title("Degree distribution")`
4. **Variable/function names are exempt** — only human-readable text

**Allowed capitals**:
- First word of a sentence or header
- Proper nouns: DRIADA, INTENSE, GCMI, KSG, PyTorch
- Class/type names when referring to code: TimeSeries, MVData, Experiment
- Acronyms: MI, PCA, AE, VAE, RSA, DR, RDM, FFT

**Examples**:
```python
# BAD:
print("Computing Feature-Feature Significance")
print("Signal Association Example Complete")
ax.set_title("Functional Network Modules")

# GOOD:
print("Computing feature-feature significance")
print("Signal association example complete")
ax.set_title("Functional network modules")
```

**Action steps**:
1. Read all print statements, docstring headers, and plot titles
2. Convert Title Case to sentence case (preserve acronyms and proper nouns)
3. Pay special attention to section headers (`[1] Some Title Here`)

**Deliverable**: List of title case violations fixed

---

## PHASE 4: EXECUTION

### 4.1 Environment Preparation
**Action**: Ensure clean execution environment
- Set matplotlib backend to non-interactive: `MPLBACKEND=Agg`
- Navigate to example directory (if needed)
- Check for any required data files

### 4.2 Run Example
**Action**: Execute with Python
```bash
cd examples/example_name/
python example_name.py
```

**Monitor**:
- Execution time
- Console output
- Any warnings or errors
- Progress indicators

**Deliverable**: Execution log (success/failure, timing, output messages)

### 4.3 Error Handling
**Action**: If errors occur
- Capture full traceback
- Identify error type (import, API change, data issue, logic error)
- Note line numbers and context

**Deliverable**: Detailed error report if applicable

---

## PHASE 5: POST-EXECUTION VALIDATION

### 5.1 File Inventory
**Action**: List all generated files
```bash
# From example directory
find . -type f -newer <timestamp_before_run>
# Or manually check output directories
```

**Check**:
- Are all files in the example's folder? (no parent/root pollution)
- Is folder depth ≤ 1 level?
- Are filenames descriptive?

**Deliverable**: Complete list of generated files with paths

### 5.2 Visual Inspection
**Action**: Read and analyze ALL generated plots/images
- Use Read tool to view each image file
- Check for:
  - **Overlapping elements** (text, legends, labels colliding)
  - **Visual consistency** (color schemes, font sizes, style)
  - **Readability** (axis labels visible, legend readable)
  - **Completeness** (all expected panels/subplots present)
  - **Professional quality** (no debug artifacts, clean layout)

**Deliverable**: Visual quality report for each plot

### 5.3 Data Output Validation
**Action**: Check any saved data files (.npy, .csv, .h5, etc.)
- Are files non-empty?
- Are formats appropriate?
- Are any metadata/headers included?

**Deliverable**: Data file validation summary

### 5.4 Scientific Meaningfulness Check (CRITICAL)
**Action**: THINK CRITICALLY about whether results make scientific sense
- **PROBLEM**: Examples can execute successfully but produce nonsense results
- **REQUIRED**: Verify results are scientifically interpretable and meaningful

**What to check**:

1. **Dimensionality Estimates** (for manifold/DR examples):
   - Do estimates match expected intrinsic dimensionality?
   - Example: Circular manifold should have dim ~1-2, not 40-60
   - **RED FLAG**: High dimensionality when low expected
   - **FIX**: Add shuffle comparison to prove structure is real

2. **Visualization Content**:
   - Does each plot show meaningful information?
   - **RED FLAG**: "Feature 2" when only 1 feature exists
   - **RED FLAG**: Empty panels or meaningless axes
   - **FIX**: Remove confusing visualizations, add meaningful ones

3. **Metric Values**:
   - Are correlations/accuracies in expected ranges?
   - Example: Circular reconstruction r > 0.85 is "Good"
   - **RED FLAG**: Perfect scores (1.0) suggest bugs
   - **RED FLAG**: Near-zero scores suggest failure

4. **Comparison Baselines**:
   - Are results compared to meaningful baselines?
   - **BEST PRACTICE**: Compare to shuffled/null hypothesis
   - **RED FLAG**: No baseline to interpret results against
   - **FIX**: Add shuffle controls, random baselines

5. **Interpretation Validity**:
   - Do conclusions follow from results?
   - Example: "Low dimensionality" with dim=73 is misleading
   - **FIX**: Add context (shuffle comparison, benchmark)

**Critical questions to ask**:
- "Would a scientist reading this learn something meaningful?"
- "Are results compared to appropriate null hypothesis?"
- "Do visualization choices help understanding or confuse?"
- "Are metric values interpretable without additional context?"

**Action steps**:
1. Read execution output carefully - do numbers make sense?
2. Check visualizations - does each plot serve a purpose?
3. Identify RED FLAGS from list above
4. Propose fixes for any meaningfulness issues found
5. DO NOT proceed to validation without addressing issues

**Deliverable**: Scientific meaningfulness assessment with issues identified

---

## PHASE 6: REPORTING

⚠️ **CRITICAL CHECKPOINT** ⚠️
**BEFORE updating any status files:**
1. Present validation results to user (see section 6.4)
2. Wait for explicit user approval
3. DO NOT auto-update EXAMPLE_VALIDATION_STATUS.md
4. DO NOT mark example as DONE without user permission

---

### 6.1 Success Report
**Action**: If example passes all checks, document:
```markdown
## Example: [name]

**Status**: ✅ PASSED

**Execution Time**: X.X seconds

**Generated Files**:
- `examples/example_name/output1.png` - Description
- `examples/example_name/output2.npy` - Description
- `examples/example_name/plots/figure1.pdf` - Description

**Visual Quality**: All plots clear, no overlaps, consistent styling

**User Experience**: Clear workflow, helpful output messages

**API Status**: Uses current API, no deprecations
```

### 6.2 Issue Report
**Action**: If issues found, document clearly:
```markdown
## Example: [name]

**Status**: ⚠️ ISSUES FOUND

**Issues**:
1. **File Organization** (Priority: HIGH)
   - File `output.png` written to root directory
   - Should be: `examples/example_name/output.png`

2. **Hardcoded Results** (Priority: MEDIUM)
   - Line 45: Comment claims "MI = 2.5 bits"
   - Should use print statement instead

3. **Visual Issues** (Priority: MEDIUM)
   - Plot1.png: Y-axis label overlaps with tick labels
   - Plot2.png: Legend outside plot boundary

4. **API Deprecation** (Priority: HIGH)
   - Line 23: Uses old `Experiment.calculate_mi()`
   - Should use: `Experiment.intense.calculate_mi()`

**Recommended Fixes**: [detailed fix suggestions]
```

### 6.3 Update Todo List
**Action**: Mark example as completed in todo list
- Use TodoWrite to update status
- Include brief note about pass/fail status

### 6.4 User Validation (CRITICAL - MANDATORY)

⚠️ **STOP: DO NOT PROCEED WITHOUT USER APPROVAL** ⚠️

**REQUIRED ACTION**: Present validation results to user and WAIT for explicit approval

**What to present**:
1. Example name and execution status (passed/failed)
2. Fixes applied (if any)
3. Execution output (condensed)
4. Generated files list
5. Key metrics/results from example output

**What user validates**:
- **Meaningfulness**: Do results make scientific sense?
- **Global picture**: Does example fit within DRIADA's goals?
- **Quality**: Are outputs publication-ready?

**CRITICAL RULES**:
- ❌ **NEVER auto-update EXAMPLE_VALIDATION_STATUS.md without user approval**
- ❌ **NEVER mark example as DONE in status file without user saying "validated" or "approved"**
- ❌ **NEVER move to next example without user explicitly approving current one**
- ✅ **ALWAYS wait for user response before updating status**
- ✅ **ASK**: "Should I mark this as validated?" or similar

**Deliverable**: User explicitly says "validated", "approved", "mark it as validated", or similar approval phrase

---

## CHECKLIST SUMMARY

For each example, verify:

- [ ] **Phase 1**: Explored relevant API modules before analysis
- [ ] **Phase 1**: Read and understood example code
- [ ] **Phase 1**: Understood synthetic data generation (if applicable)
- [ ] **Phase 2**: All outputs go to example folder (max 1 subfolder level)
- [ ] **Phase 2**: No hardcoded absolute paths
- [ ] **Phase 3**: No hardcoded results in comments
- [ ] **Phase 3**: Text explains process, prints show results
- [ ] **Phase 3**: User-friendly entry point
- [ ] **Phase 3**: Uses current API patterns
- [ ] **Phase 3**: ASCII-only characters (no Unicode symbols)
- [ ] **Phase 3**: ⚠️ CRITICAL: Searched for and removed ALL dead code (unused functions, imports)
- [ ] **Phase 3**: ⚠️ CRITICAL: No untested code in comments/prints (all shown code must execute)
- [ ] **Phase 3**: ⚠️ CRITICAL: No Title Case in headers/prints (use sentence case)
- [ ] **Phase 4**: Example executes without errors
- [ ] **Phase 5**: All generated files inventoried
- [ ] **Phase 5**: All plots visually inspected
- [ ] **Phase 5**: No overlapping elements or visual issues
- [ ] **Phase 5**: ⚠️ CRITICAL: Results make scientific sense (dimensionality, metrics, comparisons)
- [ ] **Phase 5**: ⚠️ CRITICAL: Visualizations show meaningful information (no "Feature 2" nonsense)
- [ ] **Phase 5**: ⚠️ CRITICAL: Appropriate baselines/controls present (e.g., shuffle comparison)
- [ ] **Phase 6**: Clear report of what was generated and where
- [ ] **Phase 6**: ⚠️ CRITICAL: Presented results to user and WAITED for explicit approval
- [ ] **Phase 6**: ⚠️ CRITICAL: User explicitly said "validated", "approved", or similar approval phrase
- [ ] **Phase 6**: ONLY AFTER approval: Updated EXAMPLE_VALIDATION_STATUS.md
- [ ] **Phase 6**: ONLY AFTER approval: Todo list updated

---

## EXECUTION TEMPLATE

For each example, follow this workflow:

```
1. Launch Explore agent: "Explore [relevant module] API patterns for [example context]"
2. Read example file
3. Check file organization in code
4. Check for hardcoded results
5. Check for and replace non-ASCII characters (CRITICAL - prevents runtime errors)
6. SEARCH FOR AND DELETE dead code (unused functions, imports) (CRITICAL)
7. CHECK for untested code in comments/prints - ensure all shown code executes (CRITICAL)
8. FIX Title Case in headers/prints - use sentence case (CRITICAL)
9. Run example with MPLBACKEND=Agg
9. Inventory generated files
10. Read all plots/images
11. CHECK SCIENTIFIC MEANINGFULNESS (dimensionality, metrics, baselines) (CRITICAL)
12. Generate report and present to user
13. WAIT for user validation (meaningfulness, global picture, quality)
14. Update status file only after user approval
```

---

## NOTES

- **Be thorough**: Each example represents DRIADA's public face
- **Visual quality matters**: Users will judge the package by example outputs
- **Consistency is key**: All examples should follow same organizational patterns
- **Documentation through code**: Examples should be self-explanatory
- **No assumptions**: Verify everything, even if it "should" work
