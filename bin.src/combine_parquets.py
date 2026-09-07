#!/usr/bin/env python3
"""Concatenate parquet files into one, streaming row-group by row-group.

Used by the Snakemake pipeline to combine per-chunk donut / fits / visits
tables into a single param_set-level table.

    python combine_parquets.py chunkA.parquet chunkB.parquet --output combined.parquet

Streams one row group at a time (constant memory ~ a single row group), so it
combines the large per-donut tables without loading everything into RAM —
important on the 16 GiB RSP allocation.

Inputs come from the same run_mktable but can differ between chunks (processing
drift): a chunk may lack optional columns (e.g. ``*_donut_id`` absent for some
dates) or carry a different type (``zk_CCS`` as ``list<float>`` vs ``list<double>``;
``model_flux`` as a per-stamp ``list<double>`` vs a bare scalar).  The combined
schema keeps only the columns **present in every input with a reconcilable
type** (numeric/list widths are promoted); columns absent from some input, or
with irreconcilable types (e.g. list vs scalar), are **dropped with a note**.
This guarantees a null-free, uniform output that the downstream astropy
``QTable.read`` can parse (it chokes on null strings).
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat


def promote_type(a, b):
    """Common type two columns can both be cast to (recurses into lists).

    Raises TypeError for irreconcilable types (e.g. list vs scalar, string vs
    number) so the caller can drop that column rather than null-fill it.
    """
    if a.equals(b):
        return a
    if pat.is_null(a):
        return b
    if pat.is_null(b):
        return a
    a_list = pat.is_list(a) or pat.is_large_list(a)
    b_list = pat.is_list(b) or pat.is_large_list(b)
    if a_list and b_list:
        value_t = promote_type(a.value_type, b.value_type)
        # Preserve large_list when either side is large; a 32-bit list can't
        # hold large_list offsets, so the cast would fail at write time.
        if pat.is_large_list(a) or pat.is_large_list(b):
            return pa.large_list(value_t)
        return pa.list_(value_t)
    if a_list != b_list:
        raise TypeError(f'list vs scalar: {a} vs {b}')
    if pat.is_floating(a) and pat.is_floating(b):
        return a if a.bit_width >= b.bit_width else b
    if pat.is_integer(a) and pat.is_integer(b):
        return pa.int64()
    if (pat.is_integer(a) or pat.is_floating(a)) and \
       (pat.is_integer(b) or pat.is_floating(b)):
        return pa.float64()
    raise TypeError(f'incompatible types {a} vs {b}')


def build_unified_schema(schemas):
    """Schema of columns present in **all** inputs with a reconcilable type.

    Returns ``(schema, dropped)`` where ``dropped`` maps a dropped column name
    to the reason ('absent from N/M inputs' or 'incompatible types: ...').
    """
    name_sets = [set(s.names) for s in schemas]
    common = set.intersection(*name_sets) if name_sets else set()
    order = [f.name for f in schemas[0] if f.name in common]  # first-seen order

    types, dropped = {}, {}
    for name in order:
        t = None
        try:
            for s in schemas:
                ft = s.field(name).type
                t = ft if t is None else promote_type(t, ft)
            types[name] = t
        except TypeError as e:
            dropped[name] = f'incompatible types: {e}'

    # Columns missing from any input.
    union = set().union(*name_sets) if name_sets else set()
    for name in sorted(union - common):
        n_missing = sum(name not in ns for ns in name_sets)
        dropped[name] = f'absent from {n_missing}/{len(schemas)} inputs'

    keep = [n for n in order if n in types]
    return pa.schema([pa.field(n, types[n]) for n in keep]), dropped


def conform_table(tbl, schema):
    """Reorder ``tbl`` to ``schema`` and cast columns whose type differs
    (e.g. list<float> -> list<double>).  All schema columns are present in every
    input by construction, so no null-filling is needed."""
    data = {}
    for field in schema:
        col = tbl.column(field.name)
        if not col.type.equals(field.type):
            col = col.cast(field.type)
        data[field.name] = col
    return pa.table(data, schema=schema)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', help='Input parquet files to concatenate')
    ap.add_argument('--output', required=True, help='Output combined parquet')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Drop 0-row sentinels (e.g. nights with no AOS products write an empty
    # marker so the DAG completes) BEFORE schema unification — otherwise their
    # minimal schema would shrink the common-column intersection to nothing.
    nonempty = []
    for p in args.inputs:
        pf = pq.ParquetFile(p)
        if pf.metadata.num_rows == 0:
            print(f'  note: skipping empty input {p} (0 rows)')
        else:
            nonempty.append((p, pf))
    if not nonempty:
        # Everything empty: write a valid 0-row file so the target exists.
        pf0 = pq.ParquetFile(args.inputs[0])
        pq.ParquetWriter(args.output, pf0.schema_arrow,
                         compression='snappy').close()
        print(f'combined {len(args.inputs)} files -> {args.output}: 0 rows '
              f'(all inputs empty)')
        return
    paths = [p for p, _ in nonempty]
    pfs = [pf for _, pf in nonempty]

    # Pass 1: schemas -> unified (common, reconcilable) schema.
    schemas = [pf.schema_arrow for pf in pfs]
    unified, dropped = build_unified_schema(schemas)

    for name, reason in sorted(dropped.items()):
        print(f'  note: dropped column {name!r} — {reason}')
    for field in unified:
        srctypes = {str(s.field(field.name).type) for s in schemas}
        if len(srctypes) > 1:
            print(f'  note: column {field.name!r} promoted to {field.type} '
                  f'from {sorted(srctypes)}')

    # Pass 2: stream row groups, conforming each to the unified schema.
    writer = pq.ParquetWriter(args.output, unified, compression='snappy')
    total_rows = 0
    try:
        for p, pf in zip(paths, pfs):
            for i in range(pf.num_row_groups):
                tbl = conform_table(pf.read_row_group(i), unified)
                writer.write_table(tbl)
                total_rows += tbl.num_rows
            print(f'  {p}: {pf.num_row_groups} row groups, '
                  f'{pf.metadata.num_rows} rows')
    finally:
        writer.close()

    print(f'combined {len(paths)} files -> {args.output}: {total_rows} rows '
          f'({len(unified.names)} columns, {len(dropped)} dropped)')


if __name__ == '__main__':
    main()
