# Semen analysis explorer

Streamlit app + Python module for the 10 requested analyses (age, volume, concentration, morphology vs motility,
age/volume-group summaries, distributions, correlation heat-map, missing data, categorical comparisons, data quality).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py            # interactive app (upload your own CSV in the sidebar, or use the bundled one)
python run_all.py --dpi 300     # no browser: writes all figures/tables to ./output
```

## Downloads (for your paper)
- Every figure: **PNG / PDF / SVG** (150/300/600 DPI chosen in sidebar; PDF/SVG are vector).
- Every table: **CSV / Excel / LaTeX / Markdown**.
- Top buttons: **ZIP of everything** (figures, tables, suggested captions) and one **Excel workbook of all tables**.

## Files
- `analysis.py` – all statistics and plots (each `an_*` function = one analysis)
- `export.py`   – bytes/zip helpers · `app.py` – Streamlit UI · `run_all.py` – command-line batch

## Assumptions (edit in `analysis.py` if wrong)
- No smoking column exists; `HO_Addiction` is used as the closest habit variable. Binary items coded 1 = Yes, 2 = No (`CAT_VARS`).
- Records with no ejaculate (zero-filled) and zero / "Not found" concentration are excluded from motility analyses (sidebar toggles).
- Morphology rows with normal + abnormal ≠ 100% or 0/100% values are treated as invalid for the morphology analyses.
- Tests: Spearman + Pearson; Kruskal–Wallis / Mann–Whitney. Heat-map p-values are uncorrected for multiple testing.
