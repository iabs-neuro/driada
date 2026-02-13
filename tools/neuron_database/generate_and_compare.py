"""Generate all NOF tables via configs and compare against 10.02 reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from neuron_database import (
    load_experiment, get_fully_matched_ids,
    significance_count_table, significance_fraction_table,
    significance_fraction_of_sel_table,
    mi_table, mi_table_composite,
    cross_stats_table,
    export_count_tables_excel, export_fraction_tables_excel,
    export_fraction_of_sel_tables_excel,
    export_mi_tables_excel, export_cross_stats_csv,
)

DATA_DIR = Path(__file__).parent.parent.parent / "DRIADA data" / "NOF"
REF_DIR = DATA_DIR / "NOF cross analysis 10.02.2025"
OUT_DIR = DATA_DIR / "NOF cross analysis generated"
OUT_DIR.mkdir(exist_ok=True)

# Reference feature order (from the Excel sheets)
REF_FEATURES = [
    'rest', 'speed', 'walk', 'bodydirection_2d', 'headdirection_2d',
    'locomotion', 'walls', 'center', 'corners', 'freezing', 'rear',
    'object1', 'object2', 'object3', 'object4', 'objects',
    'place', 'any object', 'place-center', 'place-corners',
    'place-object', 'place-objects', 'place-walls',
]
COMPOSITE_LABELS = ['all', 'sel1', 'sel2', 'sel3']


def load_db():
    db = load_experiment('NOF', str(DATA_DIR))
    print(f"\nDatabase: {db}")
    db.summary()
    return db


def add_prefix(table, prefix='NOF'):
    """Add experiment prefix to mouse index to match reference format."""
    table = table.copy()
    table.index = [f'{prefix}_{m}' for m in table.index]
    table.index.name = 'mouse'
    return table


# ── Generate tables ──────────────────────────────────────────────────────

def generate_all(db):
    fully_matched = get_fully_matched_ids(db)
    configs = [
        ('at least 1 sessions  min_spec=1', None),
        ('at least 4 sessions  min_spec=1', fully_matched),
    ]

    for label, matched in configs:
        print(f"\n{'='*60}")
        print(f"Generating: {label}")
        print(f"{'='*60}")

        export_count_tables_excel(
            db, OUT_DIR / f'NOF counts {label}.xlsx',
            matched_ids_per_mouse=matched, features=REF_FEATURES)

        export_fraction_tables_excel(
            db, OUT_DIR / f'NOF fractions {label}.xlsx',
            matched_ids_per_mouse=matched, features=REF_FEATURES)

        export_fraction_of_sel_tables_excel(
            db, OUT_DIR / f'NOF fractions of sel {label}.xlsx',
            matched_ids_per_mouse=matched, features=REF_FEATURES)

        export_mi_tables_excel(
            db, OUT_DIR / f'NOF MI {label}.xlsx',
            matched_ids_per_mouse=matched, features=REF_FEATURES)

        export_cross_stats_csv(
            db, OUT_DIR / f'NOF cross-stats {label}.csv',
            min_sessions=1 if '1 sessions' in label else 4)

    print(f"\nAll tables written to: {OUT_DIR}")


# ── Compare counts ───────────────────────────────────────────────────────

def compare_counts(db):
    print(f"\n{'='*60}")
    print("COMPARING COUNT TABLES")
    print(f"{'='*60}")

    fully_matched = get_fully_matched_ids(db)
    configs = [
        ('at least 1 sessions  min_spec=1', None),
        ('at least 4 sessions  min_spec=1', fully_matched),
    ]

    issues = []
    for label, matched in configs:
        ref_path = REF_DIR / f'NOF counts {label}.xlsx'
        if not ref_path.exists():
            issues.append(f"MISSING reference: {ref_path.name}")
            continue

        ref_xls = pd.ExcelFile(ref_path)
        ref_sheets = ref_xls.sheet_names

        for sheet in ref_sheets:
            ref = pd.read_excel(ref_xls, sheet_name=sheet, index_col=0)

            if sheet in COMPOSITE_LABELS:
                from neuron_database.tables import _selectivity_counts, _sel_count_to_table
                sel = _selectivity_counts(db, matched)
                n_sel = {'all': 0, 'sel1': 1, 'sel2': 2, 'sel3': 3}[sheet]
                gen = add_prefix(_sel_count_to_table(sel, db, n_sel))
            else:
                gen = add_prefix(significance_count_table(
                    db, sheet, matched_ids_per_mouse=matched))

            # Align
            gen = gen.reindex(index=ref.index, columns=ref.columns, fill_value=0)

            diff = (gen - ref).abs()
            max_diff = diff.max().max()
            total_diff = diff.sum().sum()

            if max_diff > 0:
                tag = "MISMATCH" if max_diff > 2 else "minor"
                issues.append(
                    f"[{tag}] counts/{label}/{sheet}: "
                    f"max_diff={max_diff}, total_diff={int(total_diff)}")
                if max_diff > 2:
                    # Show cells with big differences
                    for mouse in diff.index:
                        for session in diff.columns:
                            d = diff.loc[mouse, session]
                            if d > 2:
                                issues.append(
                                    f"    {mouse} {session}: "
                                    f"gen={gen.loc[mouse, session]} "
                                    f"ref={ref.loc[mouse, session]} "
                                    f"diff={d}")

    return issues


# ── Compare fractions ────────────────────────────────────────────────────

def compare_fractions(db):
    print(f"\n{'='*60}")
    print("COMPARING FRACTION TABLES")
    print(f"{'='*60}")

    fully_matched = get_fully_matched_ids(db)
    configs = [
        ('at least 1 sessions  min_spec=1', None),
        ('at least 4 sessions  min_spec=1', fully_matched),
    ]

    issues = []
    for label, matched in configs:
        ref_path = REF_DIR / f'NOF fractions {label}.xlsx'
        if not ref_path.exists():
            issues.append(f"MISSING reference: {ref_path.name}")
            continue

        ref_xls = pd.ExcelFile(ref_path)

        for sheet in ref_xls.sheet_names[:3]:  # spot-check a few
            ref = pd.read_excel(ref_xls, sheet_name=sheet, index_col=0)
            gen = add_prefix(significance_fraction_table(
                db, sheet, matched_ids_per_mouse=matched))
            gen = gen.reindex(index=ref.index, columns=ref.columns, fill_value=0)

            # Detect scale: reference might be percentages
            ref_max = ref.max().max()
            gen_max = gen.max().max()
            if ref_max > 1.5 and gen_max < 1.5:
                scale = 'percentage'
                gen_scaled = gen * 100
            else:
                scale = '0-1'
                gen_scaled = gen

            diff = (gen_scaled.fillna(0) - ref.fillna(0)).abs()
            max_diff = diff.max().max()

            if max_diff > 0.15:
                issues.append(
                    f"[fractions] {label}/{sheet} (scale={scale}): "
                    f"max_diff={max_diff:.3f}")

    return issues


# ── Compare fractions of selective ───────────────────────────────────────

def compare_fractions_of_sel(db):
    print(f"\n{'='*60}")
    print("COMPARING FRACTION-OF-SELECTIVE TABLES")
    print(f"{'='*60}")

    fully_matched = get_fully_matched_ids(db)
    configs = [
        ('at least 1 sessions  min_spec=1', None),
        ('at least 4 sessions  min_spec=1', fully_matched),
    ]

    issues = []
    for label, matched in configs:
        ref_path = REF_DIR / f'NOF fractions of sel {label}.xlsx'
        if not ref_path.exists():
            issues.append(f"MISSING reference: {ref_path.name}")
            continue

        ref_xls = pd.ExcelFile(ref_path)

        from neuron_database.tables import _selectivity_counts, _sel_count_to_table
        sel = _selectivity_counts(db, matched)
        sel_totals = _sel_count_to_table(sel, db, 0)

        for sheet in ref_xls.sheet_names[:3]:  # spot-check
            ref = pd.read_excel(ref_xls, sheet_name=sheet, index_col=0)

            if sheet in COMPOSITE_LABELS:
                continue

            gen = add_prefix(significance_fraction_of_sel_table(
                db, sheet, add_prefix(sel_totals),
                matched_ids_per_mouse=matched))
            gen = gen.reindex(index=ref.index, columns=ref.columns, fill_value=0)

            # Detect scale
            ref_max = ref.max().max()
            gen_max = gen.max().max()
            if ref_max > 1.5 and gen_max < 1.5:
                gen_scaled = gen * 100
            else:
                gen_scaled = gen

            diff = (gen_scaled.fillna(0) - ref.fillna(0)).abs()
            max_diff = diff.max().max()

            if max_diff > 0.15:
                issues.append(
                    f"[frac_of_sel] {label}/{sheet}: max_diff={max_diff:.3f}")

    return issues


# ── Compare MI tables ────────────────────────────────────────────────────

def compare_mi(db):
    print(f"\n{'='*60}")
    print("COMPARING MI TABLES")
    print(f"{'='*60}")

    fully_matched = get_fully_matched_ids(db)
    configs = [
        ('at least 1 sessions  min_spec=1', None),
        ('at least 4 sessions  min_spec=1', fully_matched),
    ]

    issues = []
    for label, matched in configs:
        ref_path = REF_DIR / f'NOF MI {label}.xlsx'
        if not ref_path.exists():
            issues.append(f"MISSING reference: {ref_path.name}")
            continue

        ref_xls = pd.ExcelFile(ref_path)

        for sheet in ref_xls.sheet_names:
            ref = pd.read_excel(ref_xls, sheet_name=sheet)
            # Rename ref columns to match ours
            ref = ref.rename(columns={'matching_row': 'matched_id'})
            if 'Unnamed: 0' in ref.columns:
                ref = ref.drop(columns=['Unnamed: 0'])

            # Strip prefix from mouse column
            if 'mouse' in ref.columns:
                ref['mouse'] = ref['mouse'].str.replace('NOF_', '', regex=False)
                ref['mouse'] = ref['mouse'].str.replace('.csv', '', regex=False)

            if sheet in COMPOSITE_LABELS:
                n_sel = {'all': 0, 'sel1': 1, 'sel2': 2, 'sel3': 3}[sheet]
                gen = mi_table_composite(db, n_sel,
                                         matched_ids_per_mouse=matched)
            else:
                gen = mi_table(db, sheet, matched_ids_per_mouse=matched)

            # Compare row counts
            ref_n = len(ref)
            gen_n = len(gen)
            if abs(ref_n - gen_n) > 0:
                tag = "MISMATCH" if abs(ref_n - gen_n) > 5 else "minor"
                issues.append(
                    f"[{tag}] MI/{label}/{sheet}: "
                    f"row_count gen={gen_n} ref={ref_n} "
                    f"diff={gen_n - ref_n}")

            if gen.empty or ref.empty:
                continue

            # Merge on mouse+matched_id to compare MI values
            sessions = db.sessions
            mi_cols = [c for c in sessions if c in gen.columns and c in ref.columns]

            merged = gen.merge(ref, on=['mouse', 'matched_id'],
                               suffixes=('_gen', '_ref'), how='inner')

            for s in mi_cols:
                g = merged.get(f'{s}_gen', merged.get(s))
                r = merged.get(f'{s}_ref')
                if g is None or r is None:
                    continue
                diff = (g.fillna(0) - r.fillna(0)).abs()
                max_d = diff.max()
                if max_d > 0.002:
                    issues.append(
                        f"[MI value] {label}/{sheet}/{s}: max_diff={max_d:.4f}")

    return issues


# ── Compare cross-stats ─────────────────────────────────────────────────

def compare_cross_stats(db):
    print(f"\n{'='*60}")
    print("COMPARING CROSS-STATS TABLES")
    print(f"{'='*60}")

    issues = []
    configs = [
        ('at least 1 sessions', 1),
        ('at least 4 sessions', 4),
    ]

    for label, min_sessions in configs:
        ref_path = REF_DIR / f'NOF cross-stats {label}.csv'
        if not ref_path.exists():
            issues.append(f"MISSING reference: {ref_path.name}")
            continue

        ref = pd.read_csv(ref_path, index_col=0)
        gen = cross_stats_table(db, min_sessions=min_sessions)

        # Add prefix to mouse column in gen
        gen['mouse'] = 'NOF_' + gen['mouse']

        issues.append(f"\n--- cross-stats {label} ---")
        issues.append(f"  ref rows: {len(ref)}, gen rows: {len(gen)}")

        # Compare per mouse
        for mouse in sorted(set(ref['mouse'].unique()) | set(gen['mouse'].unique())):
            ref_m = ref[ref['mouse'] == mouse]
            gen_m = gen[gen['mouse'] == mouse]
            if len(ref_m) != len(gen_m):
                issues.append(
                    f"  {mouse}: ref={len(ref_m)} gen={len(gen_m)} "
                    f"diff={len(gen_m) - len(ref_m)}")

        # Compare matching_row by matching_row for each mouse
        sessions = ['1D', '2D', '3D', '4D']
        n_exact_match = 0
        n_feature_mismatch = 0
        n_missing_in_gen = 0
        n_extra_in_gen = 0

        for mouse in ref['mouse'].unique():
            ref_m = ref[ref['mouse'] == mouse].set_index('matching_row')
            gen_m = gen[gen['mouse'] == mouse].set_index('matching_row')

            common = ref_m.index.intersection(gen_m.index)
            n_missing_in_gen += len(ref_m.index.difference(gen_m.index))
            n_extra_in_gen += len(gen_m.index.difference(ref_m.index))

            for mid in common:
                for s in sessions:
                    ref_val = str(ref_m.loc[mid, s]).strip() if pd.notna(ref_m.loc[mid, s]) else ''
                    gen_val = str(gen_m.loc[mid, s]).strip() if pd.notna(gen_m.loc[mid, s]) else ''

                    # Normalize: ref uses '' for absent, gen uses ''
                    # ref uses '---' for present but not selective
                    if ref_val == gen_val:
                        n_exact_match += 1
                    elif ref_val == 'nan':
                        ref_val = ''
                        if ref_val == gen_val:
                            n_exact_match += 1
                        else:
                            n_feature_mismatch += 1
                    else:
                        n_feature_mismatch += 1

        issues.append(f"  Cell comparison on common rows:")
        issues.append(f"    exact matches: {n_exact_match}")
        issues.append(f"    feature mismatches: {n_feature_mismatch}")
        issues.append(f"    rows only in ref: {n_missing_in_gen}")
        issues.append(f"    rows only in gen: {n_extra_in_gen}")

        # Show sample mismatches
        if n_feature_mismatch > 0:
            shown = 0
            for mouse in ref['mouse'].unique():
                ref_m = ref[ref['mouse'] == mouse].set_index('matching_row')
                gen_m = gen[gen['mouse'] == mouse].set_index('matching_row')
                common = ref_m.index.intersection(gen_m.index)
                for mid in common:
                    for s in sessions:
                        rv = str(ref_m.loc[mid, s]).strip()
                        gv = str(gen_m.loc[mid, s]).strip()
                        if rv == 'nan':
                            rv = ''
                        if rv != gv and shown < 15:
                            issues.append(
                                f"    SAMPLE: {mouse} mid={mid} {s}: "
                                f"ref='{rv}' gen='{gv}'")
                            shown += 1

    return issues


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    db = load_db()

    # Generate all tables
    generate_all(db)

    # Run comparisons
    all_issues = []
    all_issues.extend(compare_counts(db))
    all_issues.extend(compare_fractions(db))
    all_issues.extend(compare_fractions_of_sel(db))
    all_issues.extend(compare_mi(db))
    all_issues.extend(compare_cross_stats(db))

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    if not all_issues:
        print("No issues found!")
    else:
        for issue in all_issues:
            print(issue)
    print(f"\nTotal issues: {len([i for i in all_issues if not i.startswith('  ') and not i.startswith('---')])}")


if __name__ == '__main__':
    main()
