"""Regenerate every paper figure + table from the cached CSVs under results/.

Output goes to ./out/plots/ at the repo root by default; pass --out-dir to
override. Skips any plot whose source CSVs are missing (with a warning).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from model_xray.plots._style import default_out_dir, repo_root


PLOTS = [
    ("exp1_id_oml.py",  "--out", "exp1_id_oml.png"),
    ("exp2_oml.py",     "--id-out", "exp2_id_oml.png"),
    ("exp2_oml.py",     "--ood-out", "exp2_ood_oml.png"),
    ("exp2_al.py",      "--id-out", "exp2_id_al.png"),
    ("exp4_new.py",     "--out", "exp4_new.png"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=default_out_dir())
    parser.add_argument("--skip-tables", action="store_true",
                        help="Skip rendering the LaTeX tables for Table 4 / Table 5.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(repo_root(), "scripts", "plots")

    seen = set()
    for script, flag, out_name in PLOTS:
        # Each script may produce multiple PNGs; we delegate by passing only the
        # relevant flag(s). Calling exp2_oml twice is fine — it's idempotent.
        cmd = [sys.executable, os.path.join(plots_dir, script),
               flag, os.path.join(args.out_dir, out_name)]
        if (script, flag) in seen:
            continue
        seen.add((script, flag))
        print("$", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! {script} returned {rc}", file=sys.stderr)

    if not args.skip_tables:
        print("\n--- LaTeX tables ---")
        for table, paper_label in (("time", "Table 4"), ("memory", "Table 5")):
            print(f"\n[{paper_label} ({table})]")
            subprocess.call([sys.executable,
                             os.path.join(plots_dir, "tables_time_memory.py"),
                             "--table", table])


if __name__ == "__main__":
    main()
