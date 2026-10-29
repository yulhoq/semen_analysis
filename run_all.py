"""Command-line version: writes every figure (PNG+PDF+SVG) and table (CSV/LaTeX/Excel) to ./output.

    python run_all.py [path/to/data.csv] [--dpi 300]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import analysis as A
import export as E

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=str(Path(__file__).parent / "data" / "Semen_analysis_cleaned_final_1.csv"))
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--out", default="output")
ap.add_argument("--cat", nargs="*", default=["HO_Addiction", "Morphology_New", "Abstinence_Group"])
a = ap.parse_args()

out = Path(a.out)
res = A.build_all(A.load_raw(a.csv), A.Config(), a.cat)
for rs in res.values():
    for r in rs:
        if r.fig is not None:
            for f in ("png", "pdf", "svg"):
                p = out / "figures" / f; p.mkdir(parents=True, exist_ok=True)
                (p / f"{r.key}.{f}").write_bytes(E.fig_bytes(r.fig, f, a.dpi))
            plt.close(r.fig)
        if r.table is not None:
            for sub, data, ext in (("csv", E.table_csv(r.table), "csv"),
                                   ("latex", E.table_latex(r.table, r.title, f"tab:{r.key}"), "tex")):
                p = out / "tables" / sub; p.mkdir(parents=True, exist_ok=True)
                (p / f"{r.key}.{ext}").write_bytes(data)
(out / "tables" / "all_tables.xlsx").write_bytes(E.all_tables_xlsx(res))
(out / "captions.txt").write_text("\n\n".join(f"{r.key}: {r.caption}" for rs in res.values() for r in rs if r.caption))
print(f"Done. Files written to {out.resolve()}")
