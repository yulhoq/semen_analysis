"""
Semen analysis – statistical analysis & publication-quality figures.

Every public ``an_*`` function returns a list of ``Result`` objects
(title, matplotlib figure and/or pandas table, caption).  The Streamlit app
(app.py) and the command line script (run_all.py) both use these functions,
so what you see in the app is exactly what is exported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --------------------------------------------------------------------------
# Style (colour-blind safe Okabe-Ito palette, readable in greyscale print)
# --------------------------------------------------------------------------
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
BLUE, ORANGE, GREEN = PALETTE[0], PALETTE[1], PALETTE[2]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------
RENAME = {
    "Amount_Semen_ml": "Volume_ml",
    "Sperm_Concentration": "Concentration",
    "Total_Sperm_Count": "Total_Count",
    "Total_Motility (%)": "Total_Motility",
    "Progressive_Motility(%)": "Progressive_Motility",
    "Non_Progressive_Motility(%)": "NonProgressive_Motility",
    "Immotile(%)": "Immotile",
    "Morphology_Normal_Form": "Normal_Morphology",
    "Abnormal": "Abnormal_Morphology",
    "Abstinence_Period_Days": "Abstinence_Days",
}
LABELS = {
    "Age": "Age (years)",
    "Volume_ml": "Semen volume (mL)",
    "Concentration": "Sperm concentration (million/mL)",
    "Total_Count": "Total sperm count (million)",
    "Total_Motility": "Total motility (%)",
    "Progressive_Motility": "Progressive motility (%)",
    "NonProgressive_Motility": "Non-progressive motility (%)",
    "Immotile": "Immotile sperm (%)",
    "Normal_Morphology": "Normal morphology (%)",
    "Abnormal_Morphology": "Abnormal morphology (%)",
    "Abstinence_Days": "Abstinence (days)",
    "PH": "Semen pH",
    "VCL": "VCL", "VSL": "VSL", "VAP": "VAP", "ALH": "ALH", "LIN": "LIN",
    "STR(%)": "STR (%)", "WOB": "WOB", "BCF": "BCF", "MAD": "MAD",
}
MOTILITY = ["Total_Motility", "Progressive_Motility", "NonProgressive_Motility", "Immotile"]
HEATMAP_VARS = ["Age", "Abstinence_Days", "Volume_ml", "Concentration", "Total_Count",
                "Total_Motility", "Progressive_Motility", "NonProgressive_Motility", "Immotile",
                "Normal_Morphology", "PH", "VCL", "VSL", "VAP", "ALH", "LIN"]

# Categorical variables available for group comparison.  Coding of the binary
# questionnaire items is assumed to be 1 = Yes, 2 = No (edit here if different).
YESNO = {1: "Yes", 2: "No"}
CAT_VARS = {
    "HO_Addiction": ("History of addiction (smoking/other)", YESNO),
    "HO_Taking_Medication": ("Taking medication", YESNO),
    "Previous_Semen_Analysis": ("Previous semen analysis", YESNO),
    "Offspring_Patient": ("Has offspring", YESNO),
    "IUD_Patient": ("IUD (as coded in dataset)", YESNO),
    "HO_Opn_pain": ("Pain history (HO_Opn_pain)", YESNO),
    "LA_Urine_Problem_Burning": ("Urinary burning", YESNO),
    "HO_Outside_Country_or_Inside_Abroad": ("Lived/travelled abroad", YESNO),
    "Morphology_New": ("Morphology category (Morphology_New: 1/2)", {1: "Category 1", 2: "Category 2"}),
    "Abstinence_Group": ("Abstinence period", None),
    "Profession_Group": ("Profession", None),
}


@dataclass
class Config:
    age_cuts: tuple = (25, 30, 35, 40)          # -> <25, 25-29, 30-34, 35-39, 40+
    volume_mode: str = "quartiles"              # "quartiles" | "clinical"
    exclude_no_sample: bool = True              # rows with no ejaculate at all (zero-filled)
    require_sperm: bool = True                  # motility is undefined when concentration = 0
    valid_morphology_only: bool = True          # drop impossible morphology entries
    corr_method: str = "spearman"               # for heatmap
    heatmap_vars: tuple = tuple(HEATMAP_VARS)


@dataclass
class Result:
    key: str
    title: str
    fig: Optional[plt.Figure] = None
    table: Optional[pd.DataFrame] = None
    caption: str = ""
    table_index: bool = False
    extra_tables: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_abstinence(v) -> float:
    if pd.isna(v):
        return np.nan
    t = str(v).strip().lower()
    if not t:
        return np.nan
    m = re.search(r"[\d.]+", t)
    if not m:
        return np.nan
    x = float(m.group())
    if "month" in t:
        return x * 30
    if "week" in t:
        return x * 7
    return x


def load_raw(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename + convert to numeric + flag problem rows (nothing is dropped here)."""
    d = raw.rename(columns=RENAME).copy()
    d["Abstinence_Days"] = d["Abstinence_Days"].map(_parse_abstinence)
    for c in ["Concentration", "Volume_ml", "PH", "Total_Count"] + MOTILITY + [
        "Normal_Morphology", "Abnormal_Morphology", "Age"]:
        d[c] = _num(d[c])
    d["Not_found_conc"] = raw["Sperm_Concentration"].astype(str).str.strip().str.lower().eq("not found")

    # Row flags
    d["No_Sample"] = d["Volume_ml"].isna() & (d[MOTILITY].fillna(0).sum(axis=1) == 0)
    d["No_Sperm"] = d["Not_found_conc"] | (d["Concentration"].fillna(0) <= 0)
    ms = d["Normal_Morphology"].fillna(0) + d["Abnormal_Morphology"].fillna(0)
    d["Morph_Valid"] = ((ms - 100).abs() <= 1) & (d["Abnormal_Morphology"] > 0) & (d["Normal_Morphology"] > 0)
    d["Morph_Valid"] &= ~d["No_Sperm"]
    return d


def analysis_frame(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = d.copy()
    if cfg.exclude_no_sample:
        x = x[~x["No_Sample"]]
    if cfg.require_sperm:
        x = x[~x["No_Sperm"]]
    x = x.copy()
    if cfg.valid_morphology_only:
        x.loc[~x["Morph_Valid"], ["Normal_Morphology", "Abnormal_Morphology"]] = np.nan
    x["Age_Group"] = make_age_groups(x["Age"], cfg.age_cuts)
    x["Volume_Group"] = make_volume_groups(x["Volume_ml"], cfg.volume_mode)
    ab = x["Abstinence_Days"]
    x["Abstinence_Group"] = pd.cut(ab, [-np.inf, 2, 5, np.inf], labels=["<=2 days", "3-5 days", ">5 days"])
    top = x["Profession"].fillna("Unknown").str.strip().value_counts()
    keep = [p for p in top.index[:5] if p != "Unknown"]
    x["Profession_Group"] = x["Profession"].fillna("Unknown").str.strip().where(
        x["Profession"].fillna("Unknown").str.strip().isin(keep), "Other/Unknown")
    return x


def make_age_groups(age: pd.Series, cuts) -> pd.Categorical:
    cuts = sorted(set(int(c) for c in cuts))
    bins = [-np.inf] + cuts + [np.inf]
    labels = [f"<{cuts[0]}"] + [f"{a}–{b - 1}" for a, b in zip(cuts[:-1], cuts[1:])] + [f"≥{cuts[-1]}"]
    return pd.cut(age, bins=bins, labels=labels, right=False)


def make_volume_groups(vol: pd.Series, mode: str) -> pd.Categorical:
    if mode == "clinical":
        # WHO 2021 lower reference limit = 1.4 mL
        bins = [-np.inf, 1.5, 3.0, 4.5, np.inf]
        labels = ["<1.5 mL", "1.5–2.9 mL", "3.0–4.4 mL", "≥4.5 mL"]
        return pd.cut(vol, bins=bins, labels=labels, right=False)
    q = vol.quantile([0, .25, .5, .75, 1]).values
    q = np.unique(q)
    if len(q) < 5:
        return pd.cut(vol, 4)
    labels = [f"Q{i + 1} ({q[i]:.1f}–{q[i + 1]:.1f} mL)" for i in range(4)]
    return pd.cut(vol, bins=q, labels=labels, include_lowest=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def corr_stats(x, y) -> dict:
    m = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    n = len(m)
    if n < 3:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    rho, ps = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])
    return dict(n=n, pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=ps)


def corr_table(d: pd.DataFrame, xcol: str, ycols: list[str]) -> pd.DataFrame:
    rows = []
    for y in ycols:
        s = corr_stats(d[xcol], d[y])
        rows.append({
            "X": LABELS.get(xcol, xcol), "Y": LABELS.get(y, y), "n": s["n"],
            "Pearson r": round(s["pearson_r"], 3), "Pearson p": s["pearson_p"],
            "Spearman ρ": round(s["spearman_rho"], 3), "Spearman p": s["spearman_p"],
        })
    t = pd.DataFrame(rows)
    for c in ["Pearson p", "Spearman p"]:
        t[c] = t[c].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    return t


def scatter_ax(ax, d, x, y, color=BLUE, title=None):
    m = d[[x, y]].dropna()
    ax.scatter(m[x], m[y], s=22, alpha=.55, color=color, edgecolor="white", linewidth=.4)
    if len(m) > 3 and m[x].nunique() > 1:
        sns.regplot(data=m, x=x, y=y, scatter=False, ax=ax, color="black",
                    line_kws=dict(lw=1.6), ci=95, truncate=True)
    s = corr_stats(m[x], m[y])
    ax.set_xlabel(LABELS.get(x, x))
    ax.set_ylabel(LABELS.get(y, y))
    txt = f"Spearman ρ = {s['spearman_rho']:.2f}, {fmt_p(s['spearman_p'])}\nPearson r = {s['pearson_r']:.2f}, {fmt_p(s['pearson_p'])}\nn = {s['n']}"
    ax.text(.02, .98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#bbbbbb", alpha=.9))
    if title:
        ax.set_title(title)
    return s


def describe(g: pd.Series) -> dict:
    g = g.dropna()
    q1, q3 = (g.quantile(.25), g.quantile(.75)) if len(g) else (np.nan, np.nan)
    return {"n": int(len(g)), "Mean": g.mean(), "SD": g.std(), "Median": g.median(),
            "Q1": q1, "Q3": q3, "IQR": q3 - q1, "Min": g.min(), "Max": g.max()}


def group_summary(d: pd.DataFrame, group: str, ycols: list[str]) -> pd.DataFrame:
    rows = []
    for y in ycols:
        for lvl, g in d.groupby(group, observed=True):
            r = describe(g[y])
            rows.append({"Variable": LABELS.get(y, y), "Group": str(lvl), **r})
    t = pd.DataFrame(rows)
    num = ["Mean", "SD", "Median", "Q1", "Q3", "IQR", "Min", "Max"]
    t[num] = t[num].round(2)
    return t


def kruskal_p(d: pd.DataFrame, group: str, y: str) -> float:
    groups = [g[y].dropna().values for _, g in d.groupby(group, observed=True) if g[y].notna().sum() >= 2]
    if len(groups) < 2:
        return np.nan
    if len(groups) == 2:
        return stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue
    return stats.kruskal(*groups).pvalue


def boxstrip(ax, d, group, y, color_cycle=PALETTE):
    order = [c for c in d[group].cat.categories if (d[group] == c).any()] if hasattr(d[group], "cat") \
        else sorted(d[group].dropna().unique())
    sns.boxplot(data=d, x=group, y=y, order=order, ax=ax, palette=color_cycle[:len(order)], width=.6,
                fliersize=0, linewidth=1, boxprops=dict(alpha=.6), hue=group, legend=False)
    sns.stripplot(data=d, x=group, y=y, order=order, ax=ax, color="black", size=2.5, alpha=.35, jitter=.2)
    counts = d.groupby(group, observed=True)[y].count()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{o}\n(n={counts.get(o, 0)})" for o in order], fontsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel(LABELS.get(y, y))
    p = kruskal_p(d, group, y)
    ax.set_title(f"{LABELS.get(y, y)}  ·  {'Mann–Whitney' if len(order) == 2 else 'Kruskal–Wallis'} {fmt_p(p)}", fontsize=10)


# --------------------------------------------------------------------------
# 1  Age vs motility
# --------------------------------------------------------------------------
def an_age_motility(d: pd.DataFrame) -> list[Result]:
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_ax(axes[0], d, "Age", "Progressive_Motility", BLUE, "Age vs progressive motility")
    scatter_ax(axes[1], d, "Age", "Total_Motility", ORANGE, "Age vs total motility")
    out.append(Result("1a_age_vs_motility_scatter", "Scatter plots: age vs motility", fig,
                      caption="Scatter plots with OLS regression line (95% CI) of age against progressive and total motility."))

    groups = [g for g in d["Age_Group"].cat.categories if (d["Age_Group"] == g).any()]
    fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups) + 1, 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 100, 11)
    for ax, g, c in zip(axes, groups, PALETTE):
        v = d.loc[d["Age_Group"] == g, "Progressive_Motility"].dropna()
        ax.hist(v, bins=bins, color=c, alpha=.8, edgecolor="white")
        ax.axvline(v.median(), color="black", ls="--", lw=1)
        ax.set_title(f"{g} y (n={len(v)})", fontsize=10)
        ax.set_xlabel("Progressive motility (%)")
    axes[0].set_ylabel("Number of men")
    fig.suptitle("Distribution of progressive motility by age group (dashed = median)", fontweight="bold")
    out.append(Result("1b_age_group_histograms", "Histograms by age group", fig,
                      caption="Histograms of progressive motility within each age group. Dashed line = group median."))

    t = corr_table(d, "Age", MOTILITY)
    out.append(Result("1c_age_motility_correlation", "Correlation: age vs motility parameters", table=t,
                      caption="Pearson and Spearman correlation coefficients between age and each motility parameter."))
    return out


# --------------------------------------------------------------------------
# 2  Age group summary
# --------------------------------------------------------------------------
def an_age_group_summary(d: pd.DataFrame) -> list[Result]:
    t = group_summary(d, "Age_Group", ["Total_Motility", "Progressive_Motility"])
    pt = {LABELS[y]: kruskal_p(d, "Age_Group", y) for y in ["Total_Motility", "Progressive_Motility"]}
    cap = "Summary statistics by age group (IQR = Q3 − Q1). Kruskal–Wallis: " + "; ".join(f"{k} {fmt_p(v)}" for k, v in pt.items()) + "."
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    boxstrip(axes[0], d, "Age_Group", "Progressive_Motility")
    boxstrip(axes[1], d, "Age_Group", "Total_Motility")
    for a in axes:
        a.set_xlabel("Age group (years)")
    return [Result("2a_age_group_summary_table", "Age-group summary table", table=t, caption=cap),
            Result("2b_age_group_boxplots", "Motility by age group", fig, caption="Box plots (median, IQR) with individual observations.")]


# --------------------------------------------------------------------------
# 3  Semen volume vs progressive motility
# --------------------------------------------------------------------------
def an_volume(d: pd.DataFrame) -> list[Result]:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    scatter_ax(ax, d, "Volume_ml", "Progressive_Motility", BLUE, "Semen volume vs progressive motility")
    t = group_summary(d, "Volume_Group", ["Progressive_Motility", "Total_Motility"])
    p = kruskal_p(d, "Volume_Group", "Progressive_Motility")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    boxstrip(ax2, d, "Volume_Group", "Progressive_Motility")
    ax2.set_xlabel("Semen volume group")
    return [Result("3a_volume_vs_progressive_scatter", "Scatter: volume vs progressive motility", fig),
            Result("3b_volume_group_summary_table", "Volume-group summary table", table=t,
                   caption=f"Summary by semen-volume group. Kruskal–Wallis for progressive motility: {fmt_p(p)}."),
            Result("3c_volume_group_boxplot", "Progressive motility by volume group", fig2)]


# --------------------------------------------------------------------------
# 4  Concentration vs motility
# --------------------------------------------------------------------------
def an_concentration(d: pd.DataFrame) -> list[Result]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_ax(axes[0], d, "Concentration", "Progressive_Motility", BLUE, "Concentration vs progressive motility")
    scatter_ax(axes[1], d, "Concentration", "Total_Motility", ORANGE, "Concentration vs total motility")
    t = corr_table(d, "Concentration", ["Progressive_Motility", "Total_Motility"])
    return [Result("4a_concentration_vs_motility_scatter", "Scatter: concentration vs motility", fig),
            Result("4b_concentration_motility_correlation", "Correlation: concentration vs motility", table=t)]


# --------------------------------------------------------------------------
# 5  Distributions
# --------------------------------------------------------------------------
def _hist(ax, v, label, color, bins=20):
    v = v.dropna()
    ax.hist(v, bins=bins, color=color, alpha=.85, edgecolor="white")
    ax.axvline(v.mean(), color="black", lw=1.2, label=f"Mean {v.mean():.1f}")
    ax.axvline(v.median(), color="black", lw=1.2, ls="--", label=f"Median {v.median():.1f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of men")
    ax.set_title(f"n = {len(v)}", fontsize=9, fontweight="normal")
    ax.legend(fontsize=8)


def an_distributions(d: pd.DataFrame) -> list[Result]:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, v, c in zip(axes.ravel(), MOTILITY, PALETTE):
        _hist(ax, d[v], LABELS[v], c, bins=np.linspace(0, 100, 21))
    out = [Result("5a_motility_distributions", "Motility distributions", fig,
                  caption="Histograms of total, progressive, non-progressive and immotile sperm (solid = mean, dashed = median).")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, v, c in zip(axes.ravel(), ["Concentration", "Total_Count", "Volume_ml", "Normal_Morphology"], PALETTE[4:]):
        _hist(ax, d[v], LABELS[v], c)
    out.append(Result("5b_concentration_morphology_distributions", "Concentration, count, volume and morphology distributions", fig))
    cols = MOTILITY + ["Concentration", "Total_Count", "Volume_ml", "Normal_Morphology", "Age"]
    rows = []
    for c in cols:
        r = describe(d[c]); r["Skewness"] = d[c].skew(); r["Variable"] = LABELS[c]; rows.append(r)
    t = pd.DataFrame(rows)[["Variable", "n", "Mean", "SD", "Median", "Q1", "Q3", "Min", "Max", "Skewness"]].round(2)
    out.append(Result("5c_descriptive_statistics", "Descriptive statistics", table=t))
    return out


# --------------------------------------------------------------------------
# 6  Morphology vs motility
# --------------------------------------------------------------------------
def an_morphology(d: pd.DataFrame) -> list[Result]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_ax(axes[0], d, "Normal_Morphology", "Progressive_Motility", BLUE, "Morphology vs progressive motility")
    scatter_ax(axes[1], d, "Normal_Morphology", "Total_Motility", ORANGE, "Morphology vs total motility")
    t = corr_table(d, "Normal_Morphology", ["Progressive_Motility", "Total_Motility"])
    return [Result("6a_morphology_vs_motility_scatter", "Scatter: normal morphology vs motility", fig,
                   caption="Only records with a valid morphology entry (normal + abnormal ≈ 100%, both > 0) are used when the sidebar option is on."),
            Result("6b_morphology_motility_correlation", "Correlation: morphology vs motility", table=t)]


# --------------------------------------------------------------------------
# 7  Missing data  (uses raw data, unfiltered)
# --------------------------------------------------------------------------
def an_missing(raw: pd.DataFrame) -> list[Result]:
    r = raw.copy()
    blank = r.apply(lambda s: s.astype(str).str.strip().isin(["", "nan", "NaN"]) if s.dtype == object or str(s.dtype).startswith("str") else s.isna())
    nf = r.apply(lambda s: s.astype(str).str.strip().str.lower().eq("not found"))
    miss = blank | nf
    t = pd.DataFrame({"Variable": r.columns, "Missing (n)": miss.sum().values,
                      "Missing (%)": (miss.mean() * 100).round(1).values,
                      "Of which 'Not found'": nf.sum().values,
                      "Observed (n)": (~miss).sum().values}).sort_values("Missing (n)", ascending=False).reset_index(drop=True)
    sub = t[t["Missing (n)"] > 0].head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(sub) + 1.5))
    ax.barh(sub["Variable"], sub["Missing (%)"], color=ORANGE, alpha=.85)
    for i, v in enumerate(sub["Missing (%)"]):
        ax.text(v + .8, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Missing values (%)")
    ax.set_xlim(0, max(100, sub["Missing (%)"].max() + 8))
    ax.set_title(f"Variables with missing values (N = {len(raw)})")
    ax.grid(axis="y", visible=False)
    return [Result("7a_missing_values_table", "Missing values per variable", table=t,
                   caption="Counts blank cells and the text 'Not found' as missing, over all records in the uploaded file."),
            Result("7b_missing_values_chart", "Missing values chart", fig)]


# --------------------------------------------------------------------------
# 8  Correlation heatmap
# --------------------------------------------------------------------------
def an_heatmap(d: pd.DataFrame, cfg: Config) -> list[Result]:
    cols = [c for c in cfg.heatmap_vars if c in d.columns]
    X = d[cols]
    method = cfg.corr_method
    C = X.corr(method=method)
    P = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            m = X[[a, b]].dropna()
            if a != b and len(m) > 3:
                P.loc[a, b] = (stats.spearmanr if method == "spearman" else stats.pearsonr)(m[a], m[b])[1]
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    ann = C.round(2).astype(str)
    for a in cols:
        for b in cols:
            p = P.loc[a, b]
            if a != b and not np.isnan(p):
                ann.loc[a, b] += "*" * int((p < .05) + (p < .01) + (p < .001))
            if a == b:
                ann.loc[a, b] = "1"
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.55 * n + 2))
    sns.heatmap(C, mask=mask, annot=ann, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
                linewidths=.5, cbar_kws=dict(shrink=.7, label=f"{method.capitalize()} correlation"),
                annot_kws=dict(size=7), ax=ax, xticklabels=[LABELS.get(c, c) for c in cols],
                yticklabels=[LABELS.get(c, c) for c in cols])
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"{method.capitalize()} correlation matrix (n = {len(d)})")
    Cr = C.round(3); Cr.index.name = "Variable"
    Pf = P.map(lambda v: np.nan if np.isnan(v) else round(v, 4)); Pf.index.name = "Variable"
    return [Result("8a_correlation_heatmap", "Correlation heatmap", fig,
                   caption="* p<0.05, ** p<0.01, *** p<0.001. Total/progressive/non-progressive/immotile are mathematically linked (they sum to ~100%), so their mutual correlations are partly structural."),
            Result("8b_correlation_matrix", "Correlation coefficients", table=Cr.reset_index(), extra_tables={"p_values": Pf.reset_index()}),
            Result("8c_correlation_pvalues", "Correlation p-values", table=Pf.reset_index())]


# --------------------------------------------------------------------------
# 9  Categorical comparisons
# --------------------------------------------------------------------------
def categorical_labels(d: pd.DataFrame, var: str) -> pd.Series:
    name, mapping = CAT_VARS[var]
    s = d[var]
    if mapping:
        s = pd.to_numeric(s, errors="coerce").map(mapping)
    s = s.astype("object").where(s.notna(), None)
    return pd.Series(pd.Categorical(s), index=d.index)


def an_categorical(d: pd.DataFrame, var: str) -> list[Result]:
    name, _ = CAT_VARS[var]
    x = d.copy()
    x["_grp"] = categorical_labels(x, var)
    x = x[x["_grp"].notna()]
    x["_grp"] = x["_grp"].astype("category")
    ys = ["Progressive_Motility", "Total_Motility"]
    t = group_summary(x, "_grp", ys).rename(columns={"Group": name})
    tests = {LABELS[y]: kruskal_p(x, "_grp", y) for y in ys}
    cap = f"{name}: " + "; ".join(f"{k} {fmt_p(v)}" for k, v in tests.items()) + \
          (" (small groups, interpret with caution)" if x["_grp"].value_counts().min() < 10 else "")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y in zip(axes, ys):
        boxstrip(ax, x, "_grp", y)
    safe = re.sub(r"\W+", "_", var).lower()
    return [Result(f"9_{safe}_summary_table", f"{name}: motility summary", table=t, caption=cap),
            Result(f"9_{safe}_boxplots", f"{name}: motility comparison", fig)]


# --------------------------------------------------------------------------
# 10  Data quality (works on full prepared data)
# --------------------------------------------------------------------------
def an_quality(raw: pd.DataFrame, d: pd.DataFrame) -> list[Result]:
    tol = 1.0
    checks = {}

    def add(name, desc, mask):
        mask = mask.fillna(False)
        checks[name] = (desc, mask)

    full_dup = raw.duplicated(keep=False)
    dup_wo_id = raw.drop(columns=[c for c in ["SL_NO"] if c in raw]).duplicated(keep=False)
    add("Duplicate rows (all columns)", "Identical records", full_dup)
    add("Duplicate rows (ignoring SL_NO)", "Same values apart from serial number", dup_wo_id)
    add("Age outside 18–70", "Implausible age", ~d["Age"].between(18, 70))
    add("Percentage outside 0–100", "Any motility/morphology % < 0 or > 100",
        pd.concat([~d[c].dropna().between(0, 100) for c in MOTILITY + ["Normal_Morphology", "Abnormal_Morphology"]], axis=1).any(axis=1))
    add("Negative values", "Volume, concentration or count < 0",
        (d[["Volume_ml", "Concentration", "Total_Count"]] < 0).any(axis=1))
    add("pH outside 6–10", "Implausible semen pH", d["PH"].notna() & ~d["PH"].between(6, 10))
    add("Volume > 10 mL or < 0.5 mL", "Extreme semen volume", d["Volume_ml"].notna() & ~d["Volume_ml"].between(0.5, 10))
    add("Concentration non-numeric ('Not found')", "Text in a numeric column", d["Not_found_conc"])
    add("No sample (zero-filled row)", "No volume recorded and all motility values = 0", d["No_Sample"])
    add("Motility: total + immotile ≠ 100%", f"|Total + Immotile − 100| > {tol}",
        ((d["Total_Motility"] + d["Immotile"] - 100).abs() > tol) & ~d["No_Sample"])
    add("Motility: progressive + non-prog ≠ total", f"|Prog + NonProg − Total| > {tol}",
        ((d["Progressive_Motility"] + d["NonProgressive_Motility"] - d["Total_Motility"]).abs() > tol) & ~d["No_Sample"])
    add("Progressive > total motility", "Progressive motility exceeds total motility", d["Progressive_Motility"] > d["Total_Motility"] + tol)
    ms = d["Normal_Morphology"] + d["Abnormal_Morphology"]
    add("Morphology: normal + abnormal ≠ 100%", f"|Normal + Abnormal − 100| > {tol}", ((ms - 100).abs() > tol) & ~d["No_Sample"])
    add("Normal morphology ≥ 50%", "Biologically very unlikely (WHO lower limit 4%)", d["Normal_Morphology"] >= 50)
    exp = d["Volume_ml"] * d["Concentration"]
    add("Total count ≠ volume × concentration", "Relative difference > 5%",
        ((exp - d["Total_Count"]).abs() / exp.replace(0, np.nan) > .05))
    add("Motility present but concentration = 0", "Motility > 0 while no sperm recorded", d["No_Sperm"] & (d["Total_Motility"] > 0))

    summ = pd.DataFrame([{"Check": k, "Records flagged": int(m.sum()), "% of records": round(m.mean() * 100, 1), "Rule": desc}
                         for k, (desc, m) in checks.items()])
    flagged = []
    for k, (_, m) in checks.items():
        for i in d.index[m]:
            flagged.append({"SL_NO": d.at[i, "SL_NO"] if "SL_NO" in d else i, "Age": d.at[i, "Age"], "Issue": k})
    ft = pd.DataFrame(flagged)
    if len(ft):
        ft = ft.groupby(["SL_NO", "Age"], as_index=False)["Issue"].agg("; ".join)

    # Unusual observations: Tukey fences (3×IQR) on key variables among valid records
    ok = d[~d["No_Sample"]]
    rows = []
    for c in ["Age", "Volume_ml", "Concentration", "Total_Count", "Total_Motility", "Progressive_Motility", "PH", "VCL"]:
        s = ok[c].dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        for i, v in s[(s < lo) | (s > hi)].items():
            rows.append({"SL_NO": d.at[i, "SL_NO"], "Variable": LABELS.get(c, c), "Value": round(v, 2),
                         "Fence (3×IQR)": f"{lo:.1f} – {hi:.1f}"})
    ut = pd.DataFrame(rows)

    sub = summ[summ["Records flagged"] > 0].sort_values("Records flagged")
    fig, ax = plt.subplots(figsize=(8, 0.4 * max(len(sub), 2) + 1.5))
    ax.barh(sub["Check"], sub["Records flagged"], color=ORANGE, alpha=.85)
    for i, v in enumerate(sub["Records flagged"]):
        ax.text(v + .3, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("Records flagged"); ax.set_title(f"Data-quality checks (N = {len(d)} records)")
    ax.grid(axis="y", visible=False)
    return [Result("10a_data_quality_summary", "Data-quality check summary", table=summ,
                   caption="A record can trigger several checks. Nothing is deleted from the file; the sidebar switches control what is excluded from analyses."),
            Result("10b_data_quality_chart", "Data-quality chart", fig),
            Result("10c_flagged_records", "Flagged records", table=ft if len(ft) else pd.DataFrame({"Info": ["No records flagged"]})),
            Result("10d_unusual_observations", "Unusual observations (extreme values)",
                   table=ut if len(ut) else pd.DataFrame({"Info": ["No extreme values"]}))]


# --------------------------------------------------------------------------
def build_all(raw: pd.DataFrame, cfg: Config | None = None, cat_vars: list[str] | None = None) -> dict[str, list[Result]]:
    cfg = cfg or Config()
    prep = prepare(raw)
    d = analysis_frame(prep, cfg)
    cat_vars = cat_vars or ["HO_Addiction", "Morphology_New", "Abstinence_Group"]
    return {
        "Age vs motility": an_age_motility(d),
        "Age-group summary": an_age_group_summary(d),
        "Semen volume vs progressive motility": an_volume(d),
        "Concentration vs motility": an_concentration(d),
        "Motility & other distributions": an_distributions(d),
        "Morphology vs motility": an_morphology(d),
        "Missing data": an_missing(raw),
        "Correlation heatmap": an_heatmap(d, cfg),
        "Categorical comparisons": [r for v in cat_vars for r in an_categorical(d, v)],
        "Data quality": an_quality(raw, prep),
    }
