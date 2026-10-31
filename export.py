"""Helpers that turn figures/tables into downloadable bytes (PNG/PDF/SVG, CSV/Excel/LaTeX)."""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from analysis import Result


def fig_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """Export figures using Matplotlib's non-interactive Agg renderer."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buf = io.BytesIO()
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(
            buf,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        return buf.getvalue()
    finally:
        buf.close()


def table_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel reads ≥ / – correctly


def table_xlsx(df: pd.DataFrame, sheet: str = "Table") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=sheet[:31], index=False)
        ws = xw.sheets[sheet[:31]]
        for i, col in enumerate(df.columns, 1):
            width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) + 2
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width, 60)
    return buf.getvalue()


def table_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> bytes:
    try:
        tex = df.to_latex(index=False, escape=True, na_rep="", float_format="%.2f",
                          caption=caption or None, label=label or None, position="htbp")
    except Exception:  # jinja2 missing etc.
        tex = df.to_csv(index=False)
    return tex.encode("utf-8")


def table_markdown(df: pd.DataFrame) -> bytes:
    try:
        return df.to_markdown(index=False).encode("utf-8")
    except Exception:
        return df.to_csv(index=False).encode("utf-8")


def all_tables_xlsx(results: dict[str, list[Result]]) -> bytes:
    buf = io.BytesIO()
    used = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for rs in results.values():
            for r in rs:
                if r.table is not None:
                    name = r.key[:31]
                    while name in used:
                        name = name[:-1] + "_"
                    used.add(name)
                    r.table.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


def zip_everything(results: dict[str, list[Result]], dpi: int = 300, formats=("png", "pdf")) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        captions = []
        for rs in results.values():
            for r in rs:
                if r.fig is not None:
                    for f in formats:
                        z.writestr(f"figures/{f}/{r.key}.{f}", fig_bytes(r.fig, f, dpi))
                if r.table is not None:
                    z.writestr(f"tables/csv/{r.key}.csv", table_csv(r.table))
                    z.writestr(f"tables/latex/{r.key}.tex", table_latex(r.table, r.title, f"tab:{r.key}"))
                if r.caption:
                    captions.append(f"{r.key}: {r.caption}")
        z.writestr("tables/all_tables.xlsx", all_tables_xlsx(results))
        z.writestr("captions.txt", "\n\n".join(captions))
    return buf.getvalue()
