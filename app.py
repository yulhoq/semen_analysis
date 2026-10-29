"""
Semen analysis explorer – Streamlit app.
Run:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analysis as A
import export as E

plt.rcParams["figure.max_open_warning"] = 0
st.set_page_config(page_title="Semen analysis explorer", page_icon="🔬", layout="wide")

DEFAULT_CSV = Path(__file__).parent / "data" / "Semen_analysis_cleaned_final_1.csv"

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🔬 Settings")
up = st.sidebar.file_uploader("Upload CSV (optional)", type="csv",
                              help="Leave empty to use the bundled dataset (same column names required).")
raw = A.load_raw(up) if up else A.load_raw(DEFAULT_CSV)

st.sidebar.subheader("Which records to analyse")
excl_ns = st.sidebar.checkbox("Exclude records with no sample (zero-filled)", True)
req_sperm = st.sidebar.checkbox("Exclude records with zero / 'Not found' concentration", True,
                                help="Motility is undefined when no sperm are present.")
valid_morph = st.sidebar.checkbox("Use only valid morphology entries", True,
                                  help="Drops rows where normal + abnormal ≠ 100% or a value is 0/100%.")

st.sidebar.subheader("Grouping")
cuts_txt = st.sidebar.text_input("Age-group cut points (years)", "25, 30, 35, 40",
                                 help="25,30,35,40 → <25, 25–29, 30–34, 35–39, ≥40")
vol_mode = st.sidebar.radio("Semen-volume groups", ["quartiles", "clinical"],
                            format_func=lambda x: "Quartiles (equal-sized)" if x == "quartiles" else "Clinical cut-offs (1.5 / 3 / 4.5 mL)")
corr_method = st.sidebar.radio("Heat-map correlation", ["spearman", "pearson"], format_func=str.capitalize)
heat_vars = st.sidebar.multiselect("Heat-map variables", A.HEATMAP_VARS, A.HEATMAP_VARS,
                                   format_func=lambda c: A.LABELS.get(c, c))
cat_vars = st.sidebar.multiselect("Categorical comparisons", list(A.CAT_VARS),
                                  ["HO_Addiction", "Morphology_New", "Abstinence_Group"],
                                  format_func=lambda c: A.CAT_VARS[c][0])

st.sidebar.subheader("Export")
dpi = st.sidebar.select_slider("Figure resolution (DPI)", [150, 300, 600], 300)

try:
    cuts = tuple(sorted({int(x) for x in cuts_txt.replace(";", ",").split(",") if x.strip()}))
    assert cuts
except Exception:
    st.sidebar.error("Enter integers separated by commas, e.g. 25, 30, 35, 40")
    cuts = (25, 30, 35, 40)

cfg = A.Config(age_cuts=cuts, volume_mode=vol_mode, exclude_no_sample=excl_ns, require_sperm=req_sperm,
               valid_morphology_only=valid_morph, corr_method=corr_method, heatmap_vars=tuple(heat_vars))


@st.cache_resource(show_spinner="Running analyses…")
def run(_raw: pd.DataFrame, raw_key: str, cfg: A.Config, cat_vars: tuple):
    return A.build_all(_raw, cfg, list(cat_vars))


@st.cache_data(show_spinner=False, max_entries=400)
def cached_fig(_fig, key: str, fmt: str, dpi: int, sig: str) -> bytes:
    return E.fig_bytes(_fig, fmt, dpi)


raw_key = f"{up.name if up else 'default'}-{len(raw)}-{int(raw.select_dtypes('number').sum().sum())}"
results = run(raw, raw_key, cfg, tuple(cat_vars))
sig = f"{raw_key}|{cfg}|{cat_vars}"

prep = A.prepare(raw)
d = A.analysis_frame(prep, cfg)

# ------------------------------------------------------------------ header
st.title("Semen analysis explorer")
st.caption("Motility, concentration, morphology and data-quality analysis · every figure and table can be downloaded for your paper.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in file", len(raw))
c2.metric("Records analysed", len(d))
c3.metric("Median age (years)", f"{d['Age'].median():.0f}")
c4.metric("Median progressive motility", f"{d['Progressive_Motility'].median():.1f}%")

with st.expander("Notes on assumptions", expanded=False):
    st.markdown(
        "- Correlations report **Spearman ρ** (robust to skewed data) and **Pearson r**; group tests are "
        "**Kruskal–Wallis** (≥3 groups) / **Mann–Whitney U** (2 groups).\n"
        "- No *smoking* column exists in this dataset; **HO_Addiction** (history of addiction) is offered as the closest "
        "patient-habit variable, with coding assumed **1 = Yes, 2 = No** (edit `CAT_VARS` in `analysis.py` if different).\n"
        "- Abstinence entered as text ('2 month') is converted to days (30 days/month).\n"
        "- Nothing is deleted from your file; the sidebar switches only control what enters the analyses. "
        "The Missing-data and Data-quality tabs always use **all** records.")

zip_col1, zip_col2, _ = st.columns([1, 1, 2])
zip_col1.download_button("⬇️ Download ALL figures + tables (ZIP)",
                         data=E.zip_everything(results, dpi), file_name="semen_analysis_outputs.zip",
                         mime="application/zip", type="primary")
zip_col2.download_button("⬇️ All tables (Excel)", data=E.all_tables_xlsx(results),
                         file_name="semen_analysis_tables.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------------------------------------------------- tabs
tabs = st.tabs(list(results.keys()))
for tab, (section, rs) in zip(tabs, results.items()):
    with tab:
        st.header(section)
        for r in rs:
            st.subheader(r.title)
            if r.fig is not None:
                st.pyplot(r.fig, clear_figure=False)
                b1, b2, b3, _ = st.columns([1, 1, 1, 3])
                for col, (fmt, mime) in zip((b1, b2, b3), [("png", "image/png"), ("pdf", "application/pdf"), ("svg", "image/svg+xml")]):
                    col.download_button(f"⬇️ {fmt.upper()}", cached_fig(r.fig, r.key, fmt, dpi, sig),
                                        file_name=f"{r.key}.{fmt}", mime=mime, key=f"{r.key}-{fmt}")
            if r.table is not None:
                st.dataframe(r.table, width="stretch", hide_index=True)
                b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 2])
                b1.download_button("⬇️ CSV", E.table_csv(r.table), f"{r.key}.csv", "text/csv", key=f"{r.key}-csv")
                b2.download_button("⬇️ Excel", E.table_xlsx(r.table, r.key), f"{r.key}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{r.key}-xlsx")
                b3.download_button("⬇️ LaTeX", E.table_latex(r.table, r.title, f"tab:{r.key}"), f"{r.key}.tex",
                                   "text/plain", key=f"{r.key}-tex")
                b4.download_button("⬇️ Markdown", E.table_markdown(r.table), f"{r.key}.md", "text/markdown", key=f"{r.key}-md")
            if r.caption:
                st.caption(f"📝 Suggested caption: {r.caption}")
            st.divider()
