# DRIADA Example Validation Protocol

This protocol ensures all examples are production-ready, user-friendly, and follow consistent standards.

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

---

## PHASE 6: REPORTING

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

### 6.4 User Validation (REQUIRED)
**Action**: WAIT for user to validate results before marking example complete
- Present summary of example execution to user
- Show key results and generated visualizations
- User checks:
  - **Meaningfulness**: Do results make scientific sense?
  - **Global picture**: Does example fit within DRIADA's goals?
  - **Quality**: Are outputs publication-ready?
- **DO NOT mark example as complete until user explicitly approves**

**Deliverable**: User approval to proceed

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
- [ ] **Phase 4**: Example executes without errors
- [ ] **Phase 5**: All generated files inventoried
- [ ] **Phase 5**: All plots visually inspected
- [ ] **Phase 5**: No overlapping elements or visual issues
- [ ] **Phase 6**: Clear report of what was generated and where
- [ ] **Phase 6**: User validated results (meaningfulness, global picture, quality)
- [ ] **Phase 6**: Todo list updated after user approval

---

## EXECUTION TEMPLATE

For each example, follow this workflow:

```
1. Launch Explore agent: "Explore [relevant module] API patterns for [example context]"
2. Read example file
3. Check file organization in code
4. Check for hardcoded results
5. Check for and replace non-ASCII characters (CRITICAL - prevents runtime errors)
6. Run example with MPLBACKEND=Agg
7. Inventory generated files
8. Read all plots/images
9. Generate report and present to user
10. WAIT for user validation (meaningfulness, global picture, quality)
11. Update status file only after user approval
```

---

## NOTES

- **Be thorough**: Each example represents DRIADA's public face
- **Visual quality matters**: Users will judge the package by example outputs
- **Consistency is key**: All examples should follow same organizational patterns
- **Documentation through code**: Examples should be self-explanatory
- **No assumptions**: Verify everything, even if it "should" work
