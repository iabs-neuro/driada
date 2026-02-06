#!/usr/bin/env python3
"""
Test Validation Protocol for DRIADA 1.0 Release.

Autonomous, read-only AST-based analyzer that classifies every test function
as GREEN / YELLOW / ORANGE / RED per severity.

Usage:
    python tools/validate_tests.py [--path tests/] [--report FILE] [--json FILE]
"""

import ast
import argparse
import io
import json
import os
import re
import sys
import textwrap
import tokenize
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

GREEN = "GREEN"
YELLOW = "YELLOW"    # Minor issues (e.g. seed hygiene)
ORANGE = "ORANGE"    # Needs human review
RED = "RED"          # Clear violation

SEVERITY_ORDER = {GREEN: 0, YELLOW: 1, ORANGE: 2, RED: 3}


@dataclass
class Finding:
    severity: str  # GREEN, YELLOW, ORANGE, RED
    rule_id: str  # e.g. "R01"
    rule_name: str  # e.g. "EXISTENCE_ONLY_TEST"
    message: str  # human-readable explanation
    confidence: str = "HIGH"  # HIGH, MEDIUM, LOW


@dataclass
class TestFunctionInfo:
    file_path: str
    function_name: str
    class_name: Optional[str]
    line_number: int

    # Fixtures / decorators
    fixture_names: list = field(default_factory=list)
    decorators: list = field(default_factory=list)
    has_parametrize: bool = False
    markers: list = field(default_factory=list)

    # Assertion counts
    assertion_count: int = 0  # Python `assert` keyword statements
    assert_func_count: int = 0  # assert_*() function calls (np.testing.assert_allclose, etc.)
    pytest_raises_count: int = 0

    # Assertion breakdown
    hasattr_assertion_count: int = 0
    isinstance_assertion_count: int = 0
    value_assertion_count: int = 0

    # Mock usage
    patch_usage_count: int = 0
    magicmock_count: int = 0
    mock_only_assertions: bool = False

    # Body analysis
    body_line_count: int = 0
    function_calls: list = field(default_factory=list)
    assigned_names: list = field(default_factory=list)
    referenced_names: set = field(default_factory=set)
    has_return: bool = False

    # Tolerances
    tolerance_values: list = field(default_factory=list)

    # Comments
    comments: list = field(default_factory=list)

    # Random usage
    random_calls: list = field(default_factory=list)
    has_seed_call: bool = False

    # Body is just fixture assignment
    is_fixture_passthrough: bool = False

    # Context flags for classification demotion
    is_visualization_context: bool = False
    is_api_guard_context: bool = False

    # AST dump for duplicate detection
    normalized_ast_dump: str = ""

    # Findings from rules
    findings: list = field(default_factory=list)
    classification: str = GREEN


@dataclass
class FileLevelInfo:
    file_path: str
    test_count: int = 0
    pytest_raises_total: int = 0
    findings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comment extraction via tokenize
# ---------------------------------------------------------------------------

class CommentExtractor:
    """Extract comments from Python source and map them to line numbers."""

    def __init__(self, source: str):
        self.comments: dict[int, str] = {}
        self._extract(source)

    def _extract(self, source: str):
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for tok_type, tok_string, start, _end, _line in tokens:
                if tok_type == tokenize.COMMENT:
                    self.comments[start[0]] = tok_string
        except tokenize.TokenError:
            pass

    def get_comments_in_range(self, start_line: int, end_line: int) -> list[tuple[int, str]]:
        return [
            (line, text)
            for line, text in self.comments.items()
            if start_line <= line <= end_line
        ]


# ---------------------------------------------------------------------------
# AST-based test extraction
# ---------------------------------------------------------------------------

# Functions known to check tolerances
TOLERANCE_FUNCTIONS = {
    "allclose", "assert_allclose", "isclose", "approx",
    "assert_array_almost_equal", "assert_almost_equal",
}

RANDOM_FUNCTIONS = {
    "randn", "rand", "randint", "random", "choice", "shuffle",
    "normal", "uniform", "permutation", "random_sample",
}

SEED_FUNCTIONS = {"seed", "default_rng", "RandomState"}

MOCK_ATTR_PATTERNS = {
    "called", "call_count", "call_args", "call_args_list",
    "assert_called", "assert_called_once", "assert_called_with",
    "assert_called_once_with", "assert_any_call", "assert_not_called",
    "return_value", "side_effect",
}


def _get_dotted_name(node) -> str:
    """Get dotted name from an AST node (e.g. 'np.testing.assert_allclose')."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parent = _get_dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return ""


def _get_func_name(node: ast.Call) -> str:
    """Get the function name from a Call node."""
    return _get_dotted_name(node.func)


def _get_end_line(node: ast.AST) -> int:
    """Get end line of an AST node."""
    if hasattr(node, "end_lineno") and node.end_lineno is not None:
        return node.end_lineno
    max_line = getattr(node, "lineno", 0)
    for child in ast.walk(node):
        child_line = getattr(child, "end_lineno", None) or getattr(child, "lineno", 0)
        if child_line > max_line:
            max_line = child_line
    return max_line


def _normalize_ast_body(body: list) -> str:
    """Normalize AST body for duplicate detection: replace all constants."""

    class ConstantNormalizer(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.Constant(value="<CONST>")

        def visit_Name(self, node):
            return node  # keep variable names

    fake_mod = ast.Module(body=body, type_ignores=[])
    normalized = ConstantNormalizer().visit(fake_mod)
    try:
        return ast.dump(normalized)
    except Exception:
        return ""


class ASTTestExtractor(ast.NodeVisitor):
    """Walk a Python file and extract TestFunctionInfo for each test function."""

    def __init__(self, file_path: str, source: str, comment_extractor: CommentExtractor):
        self.file_path = file_path
        self.source = source
        self.comment_extractor = comment_extractor
        self.tests: list[TestFunctionInfo] = []
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name.startswith("Test"):
            old_class = self._current_class
            self._current_class = node.name
            self.generic_visit(node)
            self._current_class = old_class
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name.startswith("test_"):
            return

        # Skip functions decorated with @pytest.fixture (they are fixtures, not tests)
        for dec in node.decorator_list:
            dec_name = _get_dotted_name(dec) if not isinstance(dec, ast.Call) else _get_dotted_name(dec.func)
            if dec_name and "fixture" in dec_name:
                return

        info = TestFunctionInfo(
            file_path=self.file_path,
            function_name=node.name,
            class_name=self._current_class,
            line_number=node.lineno,
        )

        # Extract fixtures (function arguments, excluding 'self')
        info.fixture_names = [
            arg.arg for arg in node.args.args if arg.arg != "self"
        ]

        # Extract decorators
        for dec in node.decorator_list:
            dec_name = _get_dotted_name(dec) if not isinstance(dec, ast.Call) else _get_dotted_name(dec.func)
            if dec_name:
                info.decorators.append(dec_name)
            if "parametrize" in dec_name:
                info.has_parametrize = True
            # Extract markers
            if "pytest.mark." in dec_name:
                marker = dec_name.split("pytest.mark.")[-1]
                info.markers.append(marker)

        # Analyze body
        self._analyze_body(node, info)

        # Extract comments in function range
        end_line = _get_end_line(node)
        info.comments = self.comment_extractor.get_comments_in_range(
            node.lineno, end_line
        )

        # Normalized AST for duplicate detection
        info.normalized_ast_dump = _normalize_ast_body(node.body)

        self.tests.append(info)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _analyze_body(self, func_node: ast.FunctionDef, info: TestFunctionInfo):
        """Analyze the body of a test function."""
        body = func_node.body

        # Count non-empty, non-docstring body lines
        info.body_line_count = self._count_body_lines(func_node)

        # Track which names are mock objects
        mock_names: set[str] = set()

        # Check if body is just fixture passthrough
        info.is_fixture_passthrough = self._is_fixture_passthrough(body, info.fixture_names)

        for node in ast.walk(func_node):
            # Count assert statements
            if isinstance(node, ast.Assert):
                info.assertion_count += 1
                self._classify_assertion(node, info, mock_names)

            # Count pytest.raises
            if isinstance(node, ast.With):
                for item in node.items:
                    call_name = ""
                    if isinstance(item.context_expr, ast.Call):
                        call_name = _get_func_name(item.context_expr)
                    if "pytest.raises" in call_name or "raises" == call_name:
                        info.pytest_raises_count += 1
                    if "pytest.warns" in call_name or "warns" == call_name:
                        info.pytest_raises_count += 1  # warns is an assertion too

                    # Track patch context managers
                    if "patch" in call_name:
                        info.patch_usage_count += 1
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            mock_names.add(item.optional_vars.id)

            # Count function calls
            if isinstance(node, ast.Call):
                func_name = _get_func_name(node)
                if func_name:
                    info.function_calls.append(func_name)

                # Count assert_*() function calls as assertions
                # (np.testing.assert_allclose, np.testing.assert_array_equal, etc.)
                leaf_func = func_name.split(".")[-1] if func_name else ""
                if leaf_func.startswith("assert_"):
                    info.assert_func_count += 1
                    info.value_assertion_count += 1

                # Detect MagicMock instantiation
                if func_name in ("MagicMock", "Mock", "unittest.mock.MagicMock",
                                 "unittest.mock.Mock"):
                    info.magicmock_count += 1

                # Detect patch as decorator (already counted by decorators)
                if "patch" in func_name and not isinstance(node, ast.With):
                    # Patch calls in expressions (non-with)
                    pass

                # Extract tolerance values
                leaf_name = func_name.split(".")[-1] if func_name else ""
                if leaf_name in TOLERANCE_FUNCTIONS:
                    tol = self._extract_tolerances(node, func_name)
                    if tol:
                        info.tolerance_values.append(tol)

                # Detect random calls
                if leaf_name in RANDOM_FUNCTIONS:
                    info.random_calls.append(func_name)

                # Detect seed calls
                if leaf_name in SEED_FUNCTIONS:
                    info.has_seed_call = True

            # Track assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        info.assigned_names.append(target.id)
                        # Track mock assignments
                        if isinstance(node.value, ast.Call):
                            call_name = _get_func_name(node.value)
                            if call_name and ("MagicMock" in call_name or "Mock" in call_name
                                              or "patch" in call_name):
                                mock_names.add(target.id)

            # Detect return statements (only at test function level, not in nested defs)
            # Handled separately below via _has_direct_return

            # Track name references (for dead code detection)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                info.referenced_names.add(node.id)

        # Detect return statements only at the test function's own scope
        # (not inside nested defs/lambdas)
        info.has_return = self._has_direct_return(func_node)

        # Detect patch decorators
        for dec in func_node.decorator_list:
            dec_name = _get_dotted_name(dec) if not isinstance(dec, ast.Call) else _get_dotted_name(dec.func)
            if dec_name and "patch" in dec_name:
                info.patch_usage_count += 1

        # Determine if assertions are mock-only
        if info.assertion_count > 0 and mock_names:
            info.mock_only_assertions = self._are_mock_only_assertions(
                func_node, mock_names, info
            )

    @staticmethod
    def _has_direct_return(func_node: ast.FunctionDef) -> bool:
        """Check if the function has a return-with-value at its own scope (not nested)."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func_node:
                continue  # ast.walk still enters children; we skip below
            if isinstance(node, ast.Return) and node.value is not None:
                # Check that this Return is NOT inside a nested function
                # by walking parents. Since ast doesn't store parents,
                # use a targeted approach: walk only direct body statements.
                pass
        # Simpler approach: iterate direct body recursively, skipping nested defs
        def _check(stmts):
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue  # skip nested functions entirely
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    return True
                # Recurse into compound statements (if/for/with/try)
                for child_stmts in _iter_child_stmts(stmt):
                    if _check(child_stmts):
                        return True
            return False

        def _iter_child_stmts(node):
            """Yield statement lists from compound statement children."""
            for field_name in ("body", "orelse", "finalbody", "handlers"):
                child = getattr(node, field_name, None)
                if isinstance(child, list):
                    yield child

        return _check(func_node.body)

    def _count_body_lines(self, func_node: ast.FunctionDef) -> int:
        """Count meaningful body lines (excluding docstring)."""
        start = func_node.lineno
        end = _get_end_line(func_node)
        lines = self.source.splitlines()[start:end]  # skip def line
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Skip docstrings (rough heuristic)
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                count += 1
        return max(count, 0)

    def _is_fixture_passthrough(self, body: list, fixture_names: list[str]) -> bool:
        """Check if body is just assigning from fixtures with no other work."""
        meaningful_stmts = [
            s for s in body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        if len(meaningful_stmts) == 0:
            return True
        if len(meaningful_stmts) == 1:
            stmt = meaningful_stmts[0]
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Name) and stmt.value.id in fixture_names:
                    return True
        return False

    def _classify_assertion(self, node: ast.Assert, info: TestFunctionInfo,
                            mock_names: set[str]):
        """Classify an assert statement."""
        test_expr = node.test

        # Check for hasattr
        if isinstance(test_expr, ast.Call):
            name = _get_func_name(test_expr)
            if name == "hasattr":
                info.hasattr_assertion_count += 1
                return
            if name == "isinstance":
                info.isinstance_assertion_count += 1
                return

        # Check for isinstance in comparisons
        if isinstance(test_expr, ast.Compare):
            # isinstance check via `type(x) == SomeType`
            if isinstance(test_expr.left, ast.Call):
                name = _get_func_name(test_expr.left)
                if name == "type":
                    info.isinstance_assertion_count += 1
                    return

        info.value_assertion_count += 1

    def _are_mock_only_assertions(self, func_node: ast.FunctionDef,
                                  mock_names: set[str],
                                  info: TestFunctionInfo) -> bool:
        """Check if ALL assertions only reference mock objects."""
        has_real_assertion = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                # Check if the assertion references any non-mock name
                refs_in_assert = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        refs_in_assert.add(child.id)
                    if isinstance(child, ast.Attribute):
                        if child.attr in MOCK_ATTR_PATTERNS:
                            continue
                        # Check if the root is a mock
                        root = child
                        while isinstance(root, ast.Attribute):
                            root = root.value
                        if isinstance(root, ast.Name) and root.id not in mock_names:
                            has_real_assertion = True
                            break

                # If no reference to non-mock objects
                non_mock_refs = refs_in_assert - mock_names - {"True", "False", "None"}
                if non_mock_refs:
                    has_real_assertion = True

        return not has_real_assertion

    def _extract_tolerances(self, node: ast.Call, func_name: str) -> Optional[dict]:
        """Extract rtol/atol from a tolerance-checking call."""
        result = {"function": func_name, "line": getattr(node, "lineno", 0)}

        for kw in node.keywords:
            if kw.arg in ("rtol", "atol", "decimal", "significant"):
                val = self._get_constant_value(kw.value)
                if val is not None:
                    result[kw.arg] = val

        # Also check positional args for assert_almost_equal(a, b, decimal=7)
        if result.get("function", "").endswith("approx"):
            # pytest.approx(expected, rel=, abs=)
            for kw in node.keywords:
                if kw.arg == "rel":
                    val = self._get_constant_value(kw.value)
                    if val is not None:
                        result["rtol"] = val
                elif kw.arg == "abs":
                    val = self._get_constant_value(kw.value)
                    if val is not None:
                        result["atol"] = val

        if len(result) > 2:  # has more than just function and line
            return result
        return None

    def _get_constant_value(self, node) -> Optional[float]:
        """Extract a numeric constant from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._get_constant_value(node.operand)
            if val is not None:
                return -val
        # Handle scientific notation via BinOp: 1e-10 is parsed as Constant directly
        # in Python 3.8+, so this should be covered.
        return None


# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------

def _total_assertions(info: TestFunctionInfo) -> int:
    """Total assertion count: assert statements + assert_*() calls + pytest.raises."""
    return info.assertion_count + info.assert_func_count + info.pytest_raises_count


class Rule:
    """Base class for validation rules."""
    rule_id: str = ""
    rule_name: str = ""

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        raise NotImplementedError

    def evaluate_file(self, file_info: FileLevelInfo,
                      tests: list[TestFunctionInfo]) -> list[Finding]:
        """Optional file-level evaluation."""
        return []

    def _finding(self, severity: str, message: str,
                 confidence: str = "HIGH") -> Finding:
        return Finding(
            severity=severity,
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            message=message,
            confidence=confidence,
        )


# Known data-only fixtures that don't perform assertions
DATA_ONLY_FIXTURES = {
    "small_experiment", "medium_experiment", "large_experiment",
    "continuous_only_experiment", "discrete_only_experiment",
    "mixed_features_experiment", "multifeature_experiment",
    "experiment_factory", "spike_reconstruction_experiment",
    "swiss_roll_data", "s_curve_data", "circular_manifold_data",
    "spatial_2d_data", "correlation_pattern", "correlated_gaussian_data",
    "simple_timeseries", "multi_timeseries",
    "base_correlated_signals_small", "base_correlated_signals_medium",
    "correlated_ts_small", "correlated_ts_medium", "correlated_ts_binarized",
    "mock_experiment", "small_visual_experiment", "visual_experiment",
    "tmp_path", "tmpdir",
}


class R01_ExistenceOnlyTest(Rule):
    rule_id = "R01"
    rule_name = "EXISTENCE_ONLY_TEST"

    @staticmethod
    def _is_import_guard(info: TestFunctionInfo) -> bool:
        """Detect API import guard tests by file name, class name, or function name."""
        if info.is_api_guard_context:
            return True
        name_lower = info.function_name.lower()
        if name_lower.startswith("test_import_"):
            return True
        if info.class_name and "import" in info.class_name.lower():
            return True
        return False

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        findings = []
        total_assertions = _total_assertions(info)

        # Fixture passthrough with no assertions
        if info.is_fixture_passthrough:
            findings.append(self._finding(
                RED,
                f"Body is only assignment from fixture, no behavior tested",
            ))
            return findings

        # Zero assertions
        if total_assertions == 0:
            # Visualization context: crash-only is acceptable
            if info.is_visualization_context and info.body_line_count > 0:
                return findings

            # Check if the test uses non-data-only fixtures that might assert
            non_data_fixtures = [
                f for f in info.fixture_names
                if f not in DATA_ONLY_FIXTURES
            ]
            if non_data_fixtures and info.body_line_count > 0:
                findings.append(self._finding(
                    ORANGE,
                    f"No assertions found; relies on fixtures {non_data_fixtures} "
                    f"which may or may not validate",
                    confidence="MEDIUM",
                ))
            elif info.body_line_count > 0:
                # Has code but no assertions -- covered by R05
                pass
            else:
                findings.append(self._finding(
                    RED,
                    "Empty test body with no assertions",
                ))
            return findings

        # All assertions are hasattr-only (no value checks from assert or assert_*() calls)
        if (info.hasattr_assertion_count > 0
                and info.hasattr_assertion_count == info.assertion_count
                and info.assert_func_count == 0
                and info.pytest_raises_count == 0):
            if self._is_import_guard(info):
                return []
            findings.append(self._finding(
                RED,
                f"All {info.assertion_count} assertions are hasattr() checks; "
                f"no actual behavior is verified",
            ))
            return findings

        # All assertions are isinstance-only (no value checks from assert or assert_*() calls)
        if (info.isinstance_assertion_count > 0
                and info.value_assertion_count == 0
                and info.hasattr_assertion_count == 0
                and info.assert_func_count == 0
                and info.pytest_raises_count == 0):
            findings.append(self._finding(
                RED,
                f"All {info.isinstance_assertion_count} assertions are type checks; "
                f"no computed values are verified",
            ))

        return findings


class R02_ExcessiveMocking(Rule):
    rule_id = "R02"
    rule_name = "EXCESSIVE_MOCKING"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        findings = []
        total_mocks = info.patch_usage_count + info.magicmock_count

        # Mock-only assertions (only check assert keyword; assert_*() calls are always value assertions)
        if info.mock_only_assertions and info.assertion_count > 0 and info.assert_func_count == 0:
            findings.append(self._finding(
                RED,
                f"All {info.assertion_count} assertions only check mock interactions "
                f"(.called, .call_count); no real values verified",
            ))
            return findings

        # High mock count
        if total_mocks >= 5:
            findings.append(self._finding(
                ORANGE,
                f"{info.patch_usage_count} patches + {info.magicmock_count} MagicMocks; "
                f"consider if real behavior is being tested",
                confidence="HIGH",
            ))
            return findings

        # High mock density
        if info.body_line_count > 0:
            density = total_mocks / info.body_line_count
            if density > 0.3 and info.patch_usage_count >= 3:
                findings.append(self._finding(
                    ORANGE,
                    f"Mock density {density:.0%} ({total_mocks} mocks in "
                    f"{info.body_line_count} lines); may be testing implementation "
                    f"rather than behavior",
                    confidence="MEDIUM",
                ))

        return findings


class R03_LoweredBarComments(Rule):
    rule_id = "R03"
    rule_name = "LOWERED_BAR_COMMENTS"

    PATTERNS = [
        re.compile(r"[Rr]educed\s+from\s+\d+", re.IGNORECASE),
        re.compile(r"[Rr]elaxed\s+(threshold|tolerance|constraint|parameters?)", re.IGNORECASE),
        re.compile(r"[Aa]llow\s+small\s+(difference|error|numerical|variation)", re.IGNORECASE),
        re.compile(r"[Ll]owered?\s+(the\s+)?bar", re.IGNORECASE),
        re.compile(r"[Ll]oosened", re.IGNORECASE),
        re.compile(r"[Dd]ecreased\s+.*\s*(threshold|tolerance)", re.IGNORECASE),
        re.compile(r"[Ee]xpected\s+so\s+we\s+lower", re.IGNORECASE),
        re.compile(r"[Ss]lightly\s+relaxed", re.IGNORECASE),
        re.compile(r"[Ii]ncreased\s+(atol|rtol|tolerance)", re.IGNORECASE),
        re.compile(r"[Ll]ower\s+(the\s+)?requirements?", re.IGNORECASE),
        re.compile(r"[Rr]educed\s+(for|to)\s+(faster|speed|quick)", re.IGNORECASE),
    ]

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        matches = []
        for line_no, comment_text in info.comments:
            for pattern in self.PATTERNS:
                if pattern.search(comment_text):
                    matches.append((line_no, comment_text.strip()))
                    break  # one match per comment line is enough

        if len(matches) >= 2:
            lines = ", ".join(f"L{m[0]}" for m in matches)
            return [self._finding(
                RED,
                f"Multiple bar-lowering comments ({lines}): "
                f"suggests test rigor was systematically compromised",
            )]
        elif len(matches) == 1:
            return [self._finding(
                ORANGE,
                f"L{matches[0][0]}: {matches[0][1][:80]}",
                confidence="HIGH",
            )]
        return []


class R04_LooseNumericalTolerance(Rule):
    rule_id = "R04"
    rule_name = "LOOSE_NUMERICAL_TOLERANCE"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        findings = []

        for tol in info.tolerance_values:
            line = tol.get("line", "?")
            func = tol.get("function", "?")

            rtol = tol.get("rtol")
            atol = tol.get("atol")

            if rtol is not None and rtol >= 0.1:
                findings.append(self._finding(
                    RED,
                    f"L{line}: {func} with rtol={rtol} (10%+); "
                    f"far too loose for numerical validation",
                ))
            elif rtol is not None and rtol >= 0.01:
                findings.append(self._finding(
                    ORANGE,
                    f"L{line}: {func} with rtol={rtol} (1-10%); "
                    f"may need justification for this tolerance",
                    confidence="MEDIUM",
                ))

            if atol is not None and atol >= 1.0:
                findings.append(self._finding(
                    RED,
                    f"L{line}: {func} with atol={atol}; "
                    f"absolute tolerance >= 1.0 likely makes assertion meaningless",
                ))
            elif atol is not None and atol >= 0.1:
                findings.append(self._finding(
                    ORANGE,
                    f"L{line}: {func} with atol={atol}; "
                    f"may need justification",
                    confidence="MEDIUM",
                ))

        return findings


class R05_CrashOnlyTest(Rule):
    rule_id = "R05"
    rule_name = "CRASH_ONLY_TEST"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        if info.is_fixture_passthrough:
            return []  # Already caught by R01

        total_assertions = _total_assertions(info)
        if total_assertions == 0 and info.body_line_count > 0 and len(info.function_calls) > 0:
            # Visualization tests: "doesn't crash" is valid for plotting functions
            if info.is_visualization_context:
                return []

            # Validation-function tests: calling check_*/validate_* without assertion
            # is valid — the test verifies the function accepts valid input
            if self._is_validation_positive_path(info):
                return []

            # Has code and function calls but no assertions
            non_data_fixtures = [
                f for f in info.fixture_names
                if f not in DATA_ONLY_FIXTURES
            ]
            if non_data_fixtures:
                return [self._finding(
                    ORANGE,
                    f"No assertions; calls functions but only verifies no exception raised. "
                    f"Uses fixtures {non_data_fixtures} which may validate.",
                    confidence="MEDIUM",
                )]
            return [self._finding(
                RED,
                f"No assertions; only verifies functions don't crash "
                f"(calls: {', '.join(set(info.function_calls[:5]))})",
            )]
        return []

    @staticmethod
    def _is_validation_positive_path(info: TestFunctionInfo) -> bool:
        """Detect 'should not raise' tests for validation/check functions."""
        # If function name contains "valid" it's likely a positive-path validation test
        name_lower = info.function_name.lower()
        if "valid" in name_lower:
            return True
        # If ANY call is to a check_* or validate_* function, the test may be
        # verifying that the function accepts valid input without raising
        for call in info.function_calls:
            leaf = call.split(".")[-1]
            if leaf.startswith(("check_", "validate_")):
                return True
        return False


class R06_DeadCodeInTests(Rule):
    rule_id = "R06"
    rule_name = "DEAD_CODE_IN_TESTS"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        findings = []

        # Unused assigned variables (not starting with _)
        for name in info.assigned_names:
            if (not name.startswith("_")
                    and name not in info.referenced_names
                    and name not in info.fixture_names):
                # Check if it's the only assignment (could be intentional discard)
                if info.assigned_names.count(name) <= 1:
                    findings.append(self._finding(
                        ORANGE,
                        f"Variable '{name}' assigned but never used",
                        confidence="MEDIUM",
                    ))

        # Return statement in test function
        if info.has_return:
            findings.append(self._finding(
                ORANGE,
                "Test function has a return statement with a value",
                confidence="HIGH",
            ))

        # Commented-out code detection
        # A comment is "code-like" if the text after '# ' is valid Python.
        # This avoids false positives from explanatory comments that happen
        # to contain '=' or '(' (e.g., "# avg_on = mean([10, 8, 9, 10]) = 9.25").
        def _is_commented_out_code(text: str) -> bool:
            """Return True if text looks like commented-out Python code."""
            text = text.lstrip("# ").strip()
            if not text:
                return False
            try:
                ast.parse(text)
                # Parsed successfully -- but single words/numbers also parse
                # (as expression statements). Reject trivial expressions that
                # are more likely prose than code.
                tree = ast.parse(text)
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    val = tree.body[0].value
                    if isinstance(val, (ast.Constant, ast.Name)):
                        return False
                return True
            except SyntaxError:
                return False

        consecutive_code_comments = 0
        for _line_no, comment_text in info.comments:
            if _is_commented_out_code(comment_text):
                consecutive_code_comments += 1
            else:
                consecutive_code_comments = 0

        if consecutive_code_comments >= 3:
            findings.append(self._finding(
                ORANGE,
                f"Found {consecutive_code_comments}+ consecutive code-like comments; "
                f"likely commented-out code",
                confidence="MEDIUM",
            ))

        return findings


class R07_AssertionDensity(Rule):
    rule_id = "R07"
    rule_name = "LOW_ASSERTION_DENSITY"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        if info.body_line_count <= 20:
            return []

        total_checks = _total_assertions(info)
        density = total_checks / info.body_line_count if info.body_line_count > 0 else 0

        if density < 0.05 and total_checks > 0:
            return [self._finding(
                ORANGE,
                f"{total_checks} assertion(s) in {info.body_line_count} lines "
                f"(density={density:.1%}); extensive setup with minimal verification",
                confidence="MEDIUM",
            )]
        return []


class R08_SeedHygiene(Rule):
    rule_id = "R08"
    rule_name = "SEED_HYGIENE"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        if not info.random_calls:
            return []

        if info.has_seed_call:
            return []

        # Check if randomness only comes from fixtures (no direct random in body)
        # If test has random calls in function_calls, it's in the body
        return [self._finding(
            YELLOW,
            f"Uses random functions ({', '.join(set(info.random_calls[:3]))}) "
            f"without setting a seed; may cause flaky tests",
            confidence="MEDIUM",
        )]


class R09_DuplicateTestLogic(Rule):
    rule_id = "R09"
    rule_name = "DUPLICATE_TEST_LOGIC"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        # This rule operates at file level; individual evaluate is a no-op
        return []

    def evaluate_file(self, file_info: FileLevelInfo,
                      tests: list[TestFunctionInfo]) -> list[Finding]:
        findings = []
        # Group by class (or None for module-level)
        by_class: dict[Optional[str], list[TestFunctionInfo]] = defaultdict(list)
        for t in tests:
            by_class[t.class_name].append(t)

        for _cls, group in by_class.items():
            non_param = [t for t in group if not t.has_parametrize]
            if len(non_param) < 2:
                continue

            # Compare AST dumps pairwise
            seen_pairs: set[tuple[str, str]] = set()
            for i, t1 in enumerate(non_param):
                if not t1.normalized_ast_dump:
                    continue
                for t2 in non_param[i + 1:]:
                    if not t2.normalized_ast_dump:
                        continue
                    pair_key = tuple(sorted([t1.function_name, t2.function_name]))
                    if pair_key in seen_pairs:
                        continue

                    if t1.normalized_ast_dump == t2.normalized_ast_dump:
                        seen_pairs.add(pair_key)
                        findings.append(Finding(
                            severity=ORANGE,
                            rule_id=self.rule_id,
                            rule_name=self.rule_name,
                            message=(
                                f"{t1.function_name} (L{t1.line_number}) and "
                                f"{t2.function_name} (L{t2.line_number}) have identical "
                                f"structure; consider @pytest.mark.parametrize"
                            ),
                            confidence="MEDIUM",
                        ))

        return findings


class R10_HardcodedPathsOrState(Rule):
    rule_id = "R10"
    rule_name = "HARDCODED_PATHS_OR_STATE"

    STATE_MUTATORS = {"os.environ", "os.chdir", "importlib.reload", "sys.path.append",
                      "sys.path.insert"}

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        findings = []

        for call in info.function_calls:
            if call in self.STATE_MUTATORS or any(call.endswith(f".{m}") for m in
                                                   ("chdir", "reload")):
                findings.append(self._finding(
                    ORANGE,
                    f"Calls {call}; may affect global state and other tests",
                    confidence="HIGH",
                ))
                break  # one finding per test is enough

        return findings


class R11_MissingNegativeTests(Rule):
    rule_id = "R11"
    rule_name = "MISSING_NEGATIVE_TESTS"

    def evaluate(self, info: TestFunctionInfo) -> list[Finding]:
        return []

    def evaluate_file(self, file_info: FileLevelInfo,
                      tests: list[TestFunctionInfo]) -> list[Finding]:
        if file_info.test_count < 5:
            return []

        if file_info.pytest_raises_total == 0:
            return [Finding(
                severity=ORANGE,
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                message=(
                    f"File has {file_info.test_count} tests but zero pytest.raises(); "
                    f"no error-path coverage"
                ),
                confidence="MEDIUM",
            )]
        return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_RULES = [
    R01_ExistenceOnlyTest(),
    R02_ExcessiveMocking(),
    R03_LoweredBarComments(),
    R04_LooseNumericalTolerance(),
    R05_CrashOnlyTest(),
    R06_DeadCodeInTests(),
    R07_AssertionDensity(),
    R08_SeedHygiene(),
    R09_DuplicateTestLogic(),
    R10_HardcodedPathsOrState(),
    R11_MissingNegativeTests(),
]


class TestValidator:
    """Orchestrates test file discovery, extraction, and validation."""

    def __init__(self, test_root: str, rules: Optional[list[str]] = None):
        self.test_root = Path(test_root)
        self.active_rules = self._filter_rules(rules)
        self.all_tests: list[TestFunctionInfo] = []
        self.file_infos: list[FileLevelInfo] = []

    def _filter_rules(self, rule_ids: Optional[list[str]]) -> list[Rule]:
        if rule_ids is None:
            return ALL_RULES
        ids = {r.strip().upper() for r in rule_ids}
        return [r for r in ALL_RULES if r.rule_id in ids]

    @staticmethod
    def _is_visualization_file(source: str, fname_lower: str, fpath_lower: str) -> bool:
        """Detect if a file is a visualization/drawing test file."""
        visual_name_hints = ("drawing", "visual", "plot", "gif")
        if any(h in fname_lower for h in visual_name_hints):
            return True
        if "visualization" in fpath_lower:
            return True
        visual_imports = ("matplotlib", "plt", "draw_", "plot_", "show_mat")
        for imp in visual_imports:
            if f"import {imp}" in source or f"from {imp}" in source:
                # Only if it's a direct plotting test file
                if "plt.close" in source or "plt.show" in source:
                    return True
        return False

    def discover_and_parse(self):
        """Find all test files and extract test function metadata."""
        test_files = sorted(self.test_root.rglob("test_*.py"))
        # Also catch *_test.py
        test_files += sorted(
            f for f in self.test_root.rglob("*_test.py") if f not in test_files
        )

        for fpath in test_files:
            try:
                source = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                print(f"  WARN: Could not read {fpath}: {e}", file=sys.stderr)
                continue

            try:
                tree = ast.parse(source, filename=str(fpath))
            except SyntaxError as e:
                print(f"  WARN: Syntax error in {fpath}: {e}", file=sys.stderr)
                continue

            comment_extractor = CommentExtractor(source)
            extractor = ASTTestExtractor(str(fpath), source, comment_extractor)
            extractor.visit(tree)

            # Detect file-level context for classification demotion
            fname_lower = fpath.name.lower()
            fpath_lower = str(fpath).lower()
            is_visual_file = self._is_visualization_file(source, fname_lower, fpath_lower)
            is_api_file = any(kw in fname_lower for kw in ("import", "api_import", "exports"))

            for t in extractor.tests:
                t.is_visualization_context = is_visual_file
                t.is_api_guard_context = is_api_file

            file_info = FileLevelInfo(file_path=str(fpath))
            file_info.test_count = len(extractor.tests)
            file_info.pytest_raises_total = sum(
                t.pytest_raises_count for t in extractor.tests
            )

            self.all_tests.extend(extractor.tests)
            self.file_infos.append(file_info)

    def validate(self):
        """Run all rules on all extracted tests."""
        # Per-test rules
        for info in self.all_tests:
            for rule in self.active_rules:
                findings = rule.evaluate(info)
                info.findings.extend(findings)

            # Classify by highest severity
            if any(f.severity == RED for f in info.findings):
                info.classification = RED
            elif any(f.severity == ORANGE for f in info.findings):
                info.classification = ORANGE
            elif any(f.severity == YELLOW for f in info.findings):
                info.classification = YELLOW
            else:
                info.classification = GREEN

        # File-level rules
        tests_by_file: dict[str, list[TestFunctionInfo]] = defaultdict(list)
        for info in self.all_tests:
            tests_by_file[info.file_path].append(info)

        for file_info in self.file_infos:
            file_tests = tests_by_file.get(file_info.file_path, [])
            for rule in self.active_rules:
                file_findings = rule.evaluate_file(file_info, file_tests)
                file_info.findings.extend(file_findings)

    def get_summary(self) -> dict:
        green = sum(1 for t in self.all_tests if t.classification == GREEN)
        yellow = sum(1 for t in self.all_tests if t.classification == YELLOW)
        orange = sum(1 for t in self.all_tests if t.classification == ORANGE)
        red = sum(1 for t in self.all_tests if t.classification == RED)
        total = len(self.all_tests)

        # Count by rule
        rule_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {RED: 0, ORANGE: 0, YELLOW: 0}
        )
        for t in self.all_tests:
            for f in t.findings:
                if f.severity in (RED, ORANGE, YELLOW):
                    rule_counts[f"{f.rule_id} {f.rule_name}"][f.severity] += 1

        # File-level findings
        for fi in self.file_infos:
            for f in fi.findings:
                if f.severity in (RED, ORANGE, YELLOW):
                    rule_counts[f"{f.rule_id} {f.rule_name}"][f.severity] += 1

        return {
            "green": green,
            "yellow": yellow,
            "orange": orange,
            "red": red,
            "total": total,
            "files": len(self.file_infos),
            "rule_counts": dict(rule_counts),
        }


# ---------------------------------------------------------------------------
# Report Writer
# ---------------------------------------------------------------------------

class ReportWriter:
    """Generate console, markdown, and JSON reports."""

    def __init__(self, validator: TestValidator, test_root: str):
        self.validator = validator
        self.test_root = Path(test_root)

    def _rel_path(self, abs_path: str) -> str:
        try:
            return str(Path(abs_path).relative_to(self.test_root.parent))
        except ValueError:
            return abs_path

    def write_console(self):
        summary = self.validator.get_summary()
        total = summary["total"]
        if total == 0:
            print("No test functions found.")
            return

        green_pct = summary["green"] / total * 100
        yellow_pct = summary["yellow"] / total * 100
        orange_pct = summary["orange"] / total * 100
        red_pct = summary["red"] / total * 100

        print()
        print("=" * 60)
        print("  DRIADA Test Validation Report")
        print("=" * 60)
        print(f"  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Files:     {summary['files']}")
        print(f"  Functions: {summary['total']}")
        print()
        print(f"  GREEN:  {summary['green']:>5}  ({green_pct:.1f}%)")
        print(f"  YELLOW: {summary['yellow']:>5}  ({yellow_pct:.1f}%)")
        print(f"  ORANGE: {summary['orange']:>5}  ({orange_pct:.1f}%)")
        print(f"  RED:    {summary['red']:>5}  ({red_pct:.1f}%)")
        print()

        if summary["rule_counts"]:
            print("  Top issues by rule:")
            # Sort by total severity count descending
            sorted_rules = sorted(
                summary["rule_counts"].items(),
                key=lambda x: (
                    x[1].get(RED, 0) * 100
                    + x[1].get(ORANGE, 0) * 10
                    + x[1].get(YELLOW, 0)
                ),
                reverse=True,
            )
            for rule_name, counts in sorted_rules:
                red_c = counts.get(RED, 0)
                orange_c = counts.get(ORANGE, 0)
                yellow_c = counts.get(YELLOW, 0)
                parts = []
                if red_c:
                    parts.append(f"{red_c} RED")
                if orange_c:
                    parts.append(f"{orange_c} ORANGE")
                if yellow_c:
                    parts.append(f"{yellow_c} YELLOW")
                print(f"    {rule_name:<40s}  {', '.join(parts)}")
            print()

        print("=" * 60)
        print()

    def write_markdown(self, output_path: str):
        lines = []
        lines.append("# DRIADA Test Validation Concerns Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Collect findings by severity grouped by rule
        findings_by_sev: dict[str, dict[str, list[tuple[str, str]]]] = {
            RED: defaultdict(list),
            ORANGE: defaultdict(list),
            YELLOW: defaultdict(list),
        }

        for t in self.validator.all_tests:
            for f in t.findings:
                if f.severity not in findings_by_sev:
                    continue
                loc = f"{self._rel_path(t.file_path)}:{t.function_name} (L{t.line_number})"
                if t.class_name:
                    loc = f"{self._rel_path(t.file_path)}:{t.class_name}.{t.function_name} (L{t.line_number})"
                findings_by_sev[f.severity][f"{f.rule_id} {f.rule_name}"].append(
                    (loc, f.message)
                )

        # File-level findings
        for fi in self.validator.file_infos:
            for f in fi.findings:
                if f.severity not in findings_by_sev:
                    continue
                loc = f"{self._rel_path(fi.file_path)} (file-level)"
                findings_by_sev[f.severity][f"{f.rule_id} {f.rule_name}"].append(
                    (loc, f.message)
                )

        section_info = [
            (RED, "RED: Immediate Action Required"),
            (ORANGE, "ORANGE: Human Review Needed"),
            (YELLOW, "YELLOW: Minor Issues"),
        ]

        for sev, heading in section_info:
            lines.append(f"## {heading}")
            lines.append("")
            by_rule = findings_by_sev[sev]
            if by_rule:
                for rule_key in sorted(by_rule.keys()):
                    entries = by_rule[rule_key]
                    lines.append(f"### {rule_key}")
                    lines.append("")
                    lines.append("| Location | Issue |")
                    lines.append("|----------|-------|")
                    for loc, msg in entries:
                        msg_clean = msg.replace("|", "\\|")
                        lines.append(f"| {loc} | {msg_clean} |")
                    lines.append("")
            else:
                lines.append(f"No {sev} findings.")
                lines.append("")

        # Summary
        summary = self.validator.get_summary()
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total functions analyzed:** {summary['total']}")
        lines.append(f"- **GREEN:** {summary['green']}")
        lines.append(f"- **YELLOW:** {summary['yellow']}")
        lines.append(f"- **ORANGE:** {summary['orange']}")
        lines.append(f"- **RED:** {summary['red']}")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    def write_json(self, output_path: str):
        summary = self.validator.get_summary()
        tests_data = []
        for t in self.validator.all_tests:
            test_entry = {
                "file": self._rel_path(t.file_path),
                "function": t.function_name,
                "class": t.class_name,
                "line": t.line_number,
                "classification": t.classification,
                "findings": [
                    {
                        "rule_id": f.rule_id,
                        "rule_name": f.rule_name,
                        "severity": f.severity,
                        "message": f.message,
                        "confidence": f.confidence,
                    }
                    for f in t.findings
                    if f.severity in (RED, ORANGE, YELLOW)
                ],
            }
            tests_data.append(test_entry)

        # File-level findings
        file_data = []
        for fi in self.validator.file_infos:
            if fi.findings:
                file_data.append({
                    "file": self._rel_path(fi.file_path),
                    "findings": [
                        {
                            "rule_id": f.rule_id,
                            "rule_name": f.rule_name,
                            "severity": f.severity,
                            "message": f.message,
                            "confidence": f.confidence,
                        }
                        for f in fi.findings
                    ],
                })

        output = {
            "generated": datetime.now().isoformat(),
            "summary": summary,
            "tests": tests_data,
            "file_level": file_data,
        }

        Path(output_path).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DRIADA Test Validation Protocol — autonomous test quality analyzer"
    )
    parser.add_argument(
        "--path", default="tests/",
        help="Root test directory (default: tests/)",
    )
    parser.add_argument(
        "--report", default="test_validation_concerns.md",
        help="Output markdown concerns report (default: test_validation_concerns.md)",
    )
    parser.add_argument(
        "--json", default="test_validation_results.json",
        dest="json_path",
        help="Output JSON results (default: test_validation_results.json)",
    )
    parser.add_argument(
        "--rules", default=None,
        help="Comma-separated rule IDs to run, e.g. R01,R04 (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Include GREEN tests in console output",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only print summary to stdout, no file output",
    )

    args = parser.parse_args()

    rule_ids = args.rules.split(",") if args.rules else None

    # Resolve path relative to script location or cwd
    test_path = Path(args.path)
    if not test_path.is_absolute():
        # Try relative to cwd first, then relative to project root
        if not test_path.exists():
            project_root = Path(__file__).resolve().parent.parent
            test_path = project_root / args.path

    if not test_path.exists():
        print(f"ERROR: Test directory not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning tests in: {test_path.resolve()}")

    validator = TestValidator(str(test_path.resolve()), rules=rule_ids)

    print("Discovering and parsing test files...")
    validator.discover_and_parse()

    print(f"Found {len(validator.all_tests)} test functions in {len(validator.file_infos)} files")
    print("Running validation rules...")
    validator.validate()

    writer = ReportWriter(validator, str(test_path.resolve()))
    writer.write_console()

    if not args.summary_only:
        writer.write_markdown(args.report)
        print(f"Concerns report written to: {args.report}")

        writer.write_json(args.json_path)
        print(f"JSON results written to: {args.json_path}")

    if args.verbose:
        print("\n--- GREEN tests ---")
        for t in validator.all_tests:
            if t.classification == GREEN:
                cls_prefix = f"{t.class_name}." if t.class_name else ""
                print(f"  {writer._rel_path(t.file_path)}:{cls_prefix}{t.function_name}")


if __name__ == "__main__":
    main()
