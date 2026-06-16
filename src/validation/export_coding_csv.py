# pattern: Imperative Shell
"""
Export the first N rows of the CCA coding template to a coder-friendly CSV.

Reads `validation/cca_coding_template.parquet` (prefix-stratified, so the first N
is a valid stratified mini-set) and writes a CSV with context columns first and
the empty coding columns last, ready to fill in a spreadsheet. Code at least the
first 500 (the MVP floor), then re-ingest and run `evaluate_cca_slice`.

Run from project root:
    uv run python -m src.validation.export_coding_csv --n 500
"""

from __future__ import annotations

import argparse

import polars as pl

import src.config as config

# Columns the coder reads (context) then fills (coding, emitted empty).
_CONTEXT = ["id", "year", "news_desk", "section_name", "sample_stratum",
            "cca_score", "headline", "lead_paragraph"]
_CODING = ["cca_event", "event_type", "us_event", "event_location"]


def main(n: int = 500, out: str | None = None) -> None:
    tmpl = pl.read_parquet(config.VALIDATION_DIR / "cca_coding_template.parquet")
    sub = tmpl.head(n)
    csv = sub.select([c for c in _CONTEXT if c in sub.columns] + _CODING)
    out_path = out or str(config.VALIDATION_DIR / f"cca_coding_first{n}.csv")
    csv.write_csv(out_path)

    print(f"Wrote {csv.height} rows x {csv.width} cols -> {out_path}")  # LOG
    print("  CODE these columns (leave context columns as-is):")  # LOG
    print("    cca_event      : true / false  (is this a collective-action event?)")  # LOG
    print("    event_type     : street / strike / boycott / conventional / lawsuit / other")  # LOG
    print("    us_event       : true / false  (did it happen in the US?)")  # LOG
    print("    event_location : US, or the country/place")  # LOG
    print(f"  score-band mix in this slice:\n"
          f"{sub['sample_stratum'].value_counts().sort('sample_stratum')}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export first-N coding CSV from the CCA template.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(n=args.n, out=args.out)
