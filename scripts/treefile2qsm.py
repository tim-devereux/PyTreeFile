#!/usr/bin/env python3
"""Convert a raycloudtools treefile into a TreeQSM-style CSV.

Each output row corresponds to a non-root segment and carries:
  - start_(x,y,z): the parent segment's tip coordinates
  - end_(x,y,z):   this segment's tip coordinates (the raw x,y,z)
  - dx, dy, dz:    normalized direction from start to end

Root segments (parent_id == -1) are dropped from the output, matching the
behavior of notebooks/TreeFile2QSM.ipynb, but their coordinates are still
used as the start of their first children.

Usage:
    python treefile2qsm.py <input.txt> [output.csv]
"""
import argparse
import csv
import math
from pathlib import Path

SEGMENT_START_COL = "x"
DEFAULT_OUTPUT_COLS = [
    "tree_id",
    "parent_id",
    "radius",
    "segment_length",
    "start_x", "start_y", "start_z",
    "end_x", "end_y", "end_z",
    "dx", "dy", "dz",
    "extension",
    "branch",
    "branch_order",
    "pos_in_branch",
]


def split_segments(line):
    """Split a treefile line into per-segment lists of string fields."""
    return [p.strip().split(",") for p in line.strip().split(", ") if p.strip()]


def parse_header(line):
    """Return (tree_attr_cols, segment_attr_cols) from the header line."""
    parts = split_segments(line.lstrip("#"))
    flat = [c.strip() for p in parts for c in p]
    if SEGMENT_START_COL not in flat:
        raise ValueError(f"Header missing '{SEGMENT_START_COL}' column: {line!r}")
    idx = flat.index(SEGMENT_START_COL)
    return flat[:idx], flat[idx:]


def normalized_direction(start, end):
    dx, dy, dz = (end[i] - start[i] for i in range(3))
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return (round(dx / mag, 5), round(dy / mag, 5), round(dz / mag, 5))


def convert(input_path, output_path=None, output_cols=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_parsed.csv")
    else:
        output_path = Path(output_path)
    output_cols = output_cols or DEFAULT_OUTPUT_COLS

    with input_path.open() as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty file: {input_path}")

    header_idx = 1 if lines[0].lstrip().startswith("#") else 0
    tree_cols, seg_cols = parse_header(lines[header_idx])
    data_lines = lines[header_idx + 1:]

    seg_idx = {name: i for i, name in enumerate(seg_cols)}
    for required in ("x", "y", "z", "parent_id"):
        if required not in seg_idx:
            raise ValueError(f"Missing required segment column: {required}")

    rows = []
    for tree_id, line in enumerate(data_lines, start=1):
        parts = split_segments(line)
        if tree_cols:
            segments_raw = parts[1:]
        else:
            segments_raw = parts

        segments = []
        for s in segments_raw:
            if len(s) != len(seg_cols):
                raise ValueError(
                    f"Tree {tree_id}: segment has {len(s)} fields, "
                    f"expected {len(seg_cols)}"
                )
            segments.append([float(v) for v in s])

        for seg in segments:
            pid = int(seg[seg_idx["parent_id"]])
            if pid == -1:
                continue  # roots are dropped from output
            parent = segments[pid]
            start = (
                parent[seg_idx["x"]],
                parent[seg_idx["y"]],
                parent[seg_idx["z"]],
            )
            end = (seg[seg_idx["x"]], seg[seg_idx["y"]], seg[seg_idx["z"]])
            dvec = normalized_direction(start, end)

            row = {name: seg[seg_idx[name]] for name in seg_cols}
            row["tree_id"] = tree_id
            row["start_x"], row["start_y"], row["start_z"] = start
            row["end_x"], row["end_y"], row["end_z"] = end
            row["dx"], row["dy"], row["dz"] = dvec
            rows.append(row)

    if not rows:
        raise ValueError("No non-root segments found.")

    cols = [c for c in output_cols if c in rows[0]]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Path to the raycloudtools treefile")
    p.add_argument(
        "output", nargs="?", help="Output CSV (default: <input>_parsed.csv)"
    )
    args = p.parse_args()
    out = convert(args.input, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
