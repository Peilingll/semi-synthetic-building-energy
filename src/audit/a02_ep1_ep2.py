"""0.2 — EP1 (energiebehoefte) x EP2 (primaire fossiele energie) structure.

Why: envelope/geometry information (what SVI could ever see) enters the NTA 8800
chain through EP1, the net energy *demand*. The registered label is a binning of
EP2, which adds the installation side (heat generator, PV, renewables share).
Anything the installations contribute is invisible to a facade image, so the
EP1 -> EP2 relation is the theoretical ceiling of an envelope-only route.

Deliverables
  * scatter EP1 x EP2 coloured by energieklasse, plus a second version coloured
    by AandeelHernieuwbareEnergie
  * R2 / Spearman of EP1 -> EP2
  * label spread at fixed EP1: how many classes a narrow EP1 slice covers
  * oracle ceiling: best possible label accuracy from EP1 alone vs EP2 alone

Scope: all four-city NTA 8800 residential certificates (not the 10k SVI pool).

Outputs: reports/tables/audit/A02_ep1_ep2.md
         reports/figures/audit/A02_ep1_ep2_{class,renewable}.{png,pdf}
"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

from src.audit import ep_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = ep_raw.REPO_ROOT
OUT = REPO_ROOT / "reports" / "tables" / "audit" / "A02_ep1_ep2.md"
FIG_DIR = REPO_ROOT / "reports" / "figures" / "audit"

# Full NTA 8800 residential class ladder, best -> worst.
LADDER = ["A++++", "A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
# Merged 7-class ladder the models actually use (stage2.features).
LADDER7 = ["A", "B", "C", "D", "E", "F", "G"]

# Tableau-10 style ramp, best (green) -> worst (red); matches notebooks/_stage3_plot.py palette family.
CLASS_COLORS = ["#1B7837", "#4DAF4A", "#A6D96A", "#D9EF8B", "#FFFFBF",
                "#FEE08B", "#FDAE61", "#F46D43", "#E45756", "#C13639", "#7F0000"]


def setup_mpl() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    logger.info("[fig] %s", FIG_DIR / f"{name}.png")


def oracle_from_scalar(x: np.ndarray, y_idx: np.ndarray, n_bins: int = 200) -> dict:
    """Upper bound on label prediction from a single scalar.

    Bin x into `n_bins` equal-count bins and predict each bin's modal class.
    No function of x alone (monotone or not, at this resolution) can beat this.
    """
    q = pd.qcut(pd.Series(x), n_bins, duplicates="drop", labels=False)
    pred = np.empty(len(x), dtype=int)
    for b in np.unique(q.dropna()):
        m = (q == b).to_numpy()
        vals, cnt = np.unique(y_idx[m], return_counts=True)
        pred[m] = vals[cnt.argmax()]
    return dict(acc=float((pred == y_idx).mean()),
                macro_f1=float(f1_score(y_idx, pred, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y_idx, pred, weights="quadratic")),
                n_bins=int(q.nunique()))


def main():
    setup_mpl()
    ep = ep_raw.load(residential_only=True, nta_only=True)
    ep = ep[ep["energieklasse"].isin(LADDER)].copy()
    ep = ep.dropna(subset=["energiebehoefte", "primaire_fossiele_energie"])
    # EP-Online contains a handful of impossible entries (EP1 up to 1.1e7,
    # EP2 up to 3.7e8 kWh/m2.yr) that destroy any moment-based statistic.
    n_all = len(ep)
    ok = ep["energiebehoefte"].between(0, 1000) & ep["primaire_fossiele_energie"].between(-300, 1500)
    n_bad = int((~ok).sum())
    ep = ep[ok].copy()
    ep["cls"] = pd.Categorical(ep["energieklasse"], categories=LADDER, ordered=True)
    ep["cls_idx"] = ep["cls"].cat.codes
    ep["cls7"] = pd.Categorical(ep_raw.merge_a_classes(ep["energieklasse"]),
                                categories=LADDER7, ordered=True)
    ep["cls7_idx"] = ep["cls7"].cat.codes

    x = ep["energiebehoefte"].to_numpy()
    y = ep["primaire_fossiele_energie"].to_numpy()

    L = ["# A02 — EP1 x EP2 structure and the envelope-only ceiling", "",
         f"Scope: all NTA 8800 residential certificates in Amsterdam / Rotterdam / "
         f"Utrecht / Delft, no dedup, n={len(ep):,} "
         f"({ep['pand_id'].nunique():,} pands). EP1 = `Energiebehoefte`, "
         f"EP2 = `PrimaireFossieleEnergie` (kWh/m2.yr).", "",
         f"Plausibility filter applied: EP1 in [0, 1000], EP2 in [-300, 1500] kWh/m2.yr — "
         f"drops {n_bad} of {n_all:,} certificates ({n_bad / n_all:.4%}). Those rows are "
         f"data-entry errors in the national register (max observed EP1 1.08e7, EP2 3.70e8, "
         f"`Compactheid` 1.4e5); they are large enough to flip any Pearson r on their own, "
         f"which is why earlier inline correlations looked erratic.", ""]

    # ---------- correlation ----------
    L += ["## EP1 -> EP2 association", "",
          "| subset | n | Pearson r | R2 (linear) | Spearman rho | rho^2 |",
          "|---|---:|---:|---:|---:|---:|"]

    def row(name, sub):
        a, b = sub["energiebehoefte"].to_numpy(), sub["primaire_fossiele_energie"].to_numpy()
        if len(a) < 50:
            return
        r = pearsonr(a, b)[0]
        rho = spearmanr(a, b)[0]
        L.append(f"| {name} | {len(a):,} | {r:.3f} | {r**2:.3f} | {rho:.3f} | {rho**2:.3f} |")

    row("all", ep)
    for t, sub in ep.groupby("gebouwtype"):
        if len(sub) >= 1000:
            row(t, sub)
    L += [""]

    # renewables share as the extra axis
    sub = ep.dropna(subset=["aandeel_hernieuwbare_energie"])
    A = np.c_[np.ones(len(sub)), sub["energiebehoefte"]]
    B = np.c_[A, sub["aandeel_hernieuwbare_energie"]]
    yy = sub["primaire_fossiele_energie"].to_numpy()

    def lin_r2(M):
        beta, *_ = np.linalg.lstsq(M, yy, rcond=None)
        res = yy - M @ beta
        return 1 - res.var() / yy.var()

    L += [f"Adding `AandeelHernieuwbareEnergie` to a linear model of EP2: "
          f"R2 {lin_r2(A):.3f} -> {lin_r2(B):.3f} (n={len(sub):,}). "
          f"The gap is installation-side information with no facade correlate.", ""]

    # ---------- label spread at fixed EP1 ----------
    for ladder, idx_col, tag in [(LADDER, "cls_idx", "11-class ladder (A++++..G)"),
                                 (LADDER7, "cls7_idx", "7-class merged (A..G, model convention)")]:
        w = 10.0
        ep["ep1_bin"] = np.floor(ep["energiebehoefte"] / w) * w
        g = ep[ep["ep1_bin"].between(0, 400)].groupby("ep1_bin")[idx_col]
        stat = pd.DataFrame({"n": g.size(), "n_cls": g.nunique(), "std": g.std(),
                             "p05": g.quantile(0.05), "p95": g.quantile(0.95),
                             "modal_share": g.apply(
                                 lambda s: s.value_counts(normalize=True).iloc[0])})
        stat = stat[stat["n"] >= 100]
        L += [f"## Label spread inside a {w:.0f} kWh/m2.yr slice of EP1 — {tag}", "",
              "| quantity | median over slices |", "|---|---:|",
              f"| certificates per slice | {stat['n'].median():.0f} |",
              f"| distinct classes present | {stat['n_cls'].median():.1f} |",
              f"| class-index span p5-p95 (steps) | {(stat['p95'] - stat['p05']).median():.1f} |",
              f"| std of class index | {stat['std'].median():.2f} |",
              f"| modal-class share (oracle acc at fixed EP1) | {stat['modal_share'].median():.3f} |",
              ""]

    # ---------- oracle ceilings ----------
    L += ["## Oracle ceiling: label from one scalar alone", "",
          "200 equal-count bins, predict each bin's modal class. Upper bound for any "
          "model whose only information is that scalar.", "",
          "| predictor | target ladder | acc | macro-F1 | quad. kappa |",
          "|---|---|---:|---:|---:|"]
    for pname, arr in [("EP1 energiebehoefte", x), ("EP2 primaire fossiele energie", y)]:
        for tag, col in [("11-class", "cls_idx"), ("7-class", "cls7_idx")]:
            o = oracle_from_scalar(arr, ep[col].to_numpy())
            L += [f"| {pname} | {tag} | {o['acc']:.3f} | {o['macro_f1']:.3f} | {o['kappa']:.3f} |"]
    L += [""]

    # ---------- same ceiling, made comparable to the experiments ----------
    # The oracle above is certificate-level, four-city, and in-sample: it is NOT
    # comparable to the reported route numbers. Recompute on the experiment pool
    # (dev, pand-level, latest certificate) with the modal label cross-fitted over
    # the same 5 folds, which is the only version that can sit next to M1/M2/M3.
    dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "dev_fold_indices.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str).str.zfill(16)
    gt = pd.read_parquet(REPO_ROOT / "data" / "processed" / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    gt["label"] = ep_raw.merge_a_classes(gt["Energieklasse"])
    latest = (ep.sort_values("reg_date").drop_duplicates("pand_id", keep="last")
              [["pand_id", "energiebehoefte"]])
    d = (dev[["pand_id", "fold"]]
         .merge(gt[["pand_id", "label", "building_type", "tabula_period"]], on="pand_id")
         .merge(latest, on="pand_id"))
    d = d[d["label"].isin(LADDER7)].reset_index(drop=True)
    d["cell"] = d["building_type"].astype(str) + "|" + d["tabula_period"].astype(str)
    yi = d["label"].map({c: i for i, c in enumerate(LADDER7)}).to_numpy()
    folds = d["fold"].to_numpy()

    def cv_oracle(key: pd.Series) -> np.ndarray:
        p = np.empty(len(d), dtype=int)
        for f in range(5):
            tr, va = folds != f, folds == f
            modal = (pd.Series(yi[tr]).groupby(key[tr].to_numpy())
                     .agg(lambda s: s.value_counts().idxmax()))
            fb = np.bincount(yi[tr]).argmax()
            p[va] = (pd.Series(key[va].to_numpy()).map(modal)
                     .fillna(fb).astype(int).to_numpy())
        return p

    L += ["## The same ceiling on the experiment pool (comparable to M1/M2/M3)", "",
          f"dev pool, pand level, latest certificate per pand, n={len(d):,}; modal label "
          f"cross-fitted over the frozen 5 folds (no in-sample optimism).", "",
          "| predictor | acc | macro-F1 | quad. kappa |", "|---|---:|---:|---:|"]
    for name, key in [("majority class", pd.Series(["_"] * len(d))),
                      ("TABULA cell", d["cell"]),
                      ("EP1, 20 equal-count bins", pd.qcut(d["energiebehoefte"], 20,
                                                           duplicates="drop", labels=False)),
                      ("EP1, 50 equal-count bins", pd.qcut(d["energiebehoefte"], 50,
                                                           duplicates="drop", labels=False)),
                      ("EP1, 200 equal-count bins", pd.qcut(d["energiebehoefte"], 200,
                                                            duplicates="drop", labels=False))]:
        p = cv_oracle(key.astype(str) if key.dtype == object else key)
        L += [f"| {name} | {(p == yi).mean():.4f} | "
              f"{f1_score(yi, p, average='macro', zero_division=0):.4f} | "
              f"{cohen_kappa_score(yi, p, weights='quadratic'):.4f} |"]
    L += ["",
          "Reference points from `reports/tables/stage3/T3_main.md` (hold-out): "
          "M0 acc 0.310 / mF1 0.068, M1 0.356 / 0.172, M2-DINOv2 0.266 / 0.213, "
          "M3-DINOv2 0.292 / 0.150.", "",
          "**Use the 50-bin row (acc ~0.64 / macro-F1 ~0.58) as the envelope-only "
          "ceiling, not the 0.709 certificate-level figure above** — that one is "
          "in-sample and computed on a different, four-city certificate population.", ""]

    # ---------- figures ----------
    rng = np.random.default_rng(0)
    samp = rng.choice(len(ep), size=min(80_000, len(ep)), replace=False)
    s = ep.iloc[samp]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, c in enumerate(LADDER):
        sc = s[s["energieklasse"] == c]
        if sc.empty:
            continue
        ax.scatter(sc["energiebehoefte"], sc["primaire_fossiele_energie"],
                   s=2, alpha=0.35, linewidths=0, color=CLASS_COLORS[i], label=c)
    ax.set_xlim(0, 500); ax.set_ylim(-100, 700)
    ax.set_xlabel("EP1  energiebehoefte  (kWh/m$^2$·yr)")
    ax.set_ylabel("EP2  primaire fossiele energie  (kWh/m$^2$·yr)")
    ax.set_title("Registered label is a binning of EP2, not of EP1")
    leg = ax.legend(title="Energieklasse", markerscale=6, ncol=2, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0))
    leg._legend_box.align = "left"
    save_fig(fig, "A02_ep1_ep2_class")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sr = s.dropna(subset=["aandeel_hernieuwbare_energie"])
    im = ax.scatter(sr["energiebehoefte"], sr["primaire_fossiele_energie"],
                    c=sr["aandeel_hernieuwbare_energie"].clip(0, 100),
                    s=2, alpha=0.4, linewidths=0, cmap="viridis")
    ax.set_xlim(0, 500); ax.set_ylim(-100, 700)
    ax.set_xlabel("EP1  energiebehoefte  (kWh/m$^2$·yr)")
    ax.set_ylabel("EP2  primaire fossiele energie  (kWh/m$^2$·yr)")
    ax.set_title("Vertical spread at fixed EP1 = installation side")
    fig.colorbar(im, ax=ax, label="aandeel hernieuwbare energie (%)")
    save_fig(fig, "A02_ep1_ep2_renewable")
    plt.close(fig)

    L += ["## Figures", "",
          "- `reports/figures/audit/A02_ep1_ep2_class.png`",
          "- `reports/figures/audit/A02_ep1_ep2_renewable.png`", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    mpl.use("Agg")
    main()
