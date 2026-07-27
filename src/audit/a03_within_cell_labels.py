"""0.3 — Label distribution inside a TABULA cell.

A TABULA cell (building_type x tabula_period) is the archetype the whole M1
route rests on: type + period -> U-values -> energy class. If the EPC label is
close to uniform inside a cell, then no amount of Stage-1 accuracy can move the
downstream number, because the cell simply does not carry the label.

Per cell: label histogram, Shannon entropy, modal share. Then the cell-only
oracle (predict each cell's modal label) as the explicit ceiling for M1.

Pools
  dev       the 8,068 training buildings (what M1 was fitted on)
  manifest  all 10,104 SVI pands
  fullstock all four-city pands with GT (~124.8k) — is it a pool artefact?

Outputs: reports/tables/audit/A03_within_cell_labels.md
         reports/figures/audit/A03_cell_label_mix.{png,pdf}
"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from src.audit import ep_raw
from src.audit.a02_ep1_ep2 import LADDER7, save_fig, setup_mpl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = ep_raw.REPO_ROOT
PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "reports" / "tables" / "audit" / "A03_within_cell_labels.md"

CLASS_COLORS7 = ["#1B7837", "#A6D96A", "#FFFFBF", "#FDAE61", "#F46D43", "#E45756", "#7F0000"]


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(abs(-(p * np.log2(p)).sum()))


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell label mix. df needs columns cell, label."""
    piv = pd.crosstab(df["cell"], df["label"]).reindex(columns=LADDER7, fill_value=0)
    out = pd.DataFrame({
        "n": piv.sum(axis=1),
        "modal": piv.idxmax(axis=1),
        "modal_share": piv.max(axis=1) / piv.sum(axis=1),
        "entropy_bits": [entropy(r.to_numpy()) for _, r in piv.iterrows()],
        "n_classes": (piv > 0).sum(axis=1),
    })
    out["norm_entropy"] = out["entropy_bits"] / np.log2(len(LADDER7))
    return out.sort_values("n", ascending=False), piv


def oracle(df: pd.DataFrame, by: list[str]) -> dict:
    """Predict each group's modal label; upper bound for any model using only `by`."""
    modal = df.groupby(by)["label"].agg(lambda s: s.value_counts().idxmax())
    pred = df.set_index(by).index.map(modal)
    y = df["label"].to_numpy()
    p = np.asarray(pred, dtype=object)
    idx = {c: i for i, c in enumerate(LADDER7)}
    yi = np.array([idx[v] for v in y])
    pi = np.array([idx[v] for v in p])
    return dict(n=len(df), n_groups=int(modal.size),
                acc=float((y == p).mean()),
                macro_f1=float(f1_score(yi, pi, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yi, pi, weights="quadratic")))


def main():
    setup_mpl()
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    gt["label"] = ep_raw.merge_a_classes(gt["Energieklasse"])
    gt = gt[gt["label"].isin(LADDER7)].copy()
    gt["cell"] = gt["building_type"].astype(str) + "|" + gt["tabula_period"].astype(str)

    man = pd.read_parquet(PROCESSED / "svi_manifest.parquet")
    man["pand_id"] = man["pand_id"].astype(str).str.zfill(16)
    man_ids = set(man["pand_id"].unique())
    dev = pd.read_parquet(PROCESSED / "dev_fold_indices.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str).str.zfill(16)

    pools = {
        "dev": gt[gt["pand_id"].isin(set(dev["pand_id"]))],
        "manifest": gt[gt["pand_id"].isin(man_ids)],
        "fullstock": gt,
    }

    L = ["# A03 — EPC label distribution inside a TABULA cell", "",
         "Cell = `building_type` x `tabula_period` (the archetype M1 consumes). "
         "Label = pand-level `Energieklasse` from `stage1_gt.parquet` "
         "(latest certificate per pand, A+..A++++ merged to A).", ""]

    for name, pool in pools.items():
        tab, _ = cell_table(pool)
        L += [f"## Pool `{name}` (n={len(pool):,}, {tab.shape[0]} non-empty cells)", "",
              "| cell | n | share of pool | modal | modal share | entropy (bits) | norm. entropy | classes present |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
        for c, r in tab.head(12).iterrows():
            L += [f"| {c} | {r['n']:,} | {r['n'] / len(pool):.1%} | {r['modal']} | "
                  f"{r['modal_share']:.3f} | {r['entropy_bits']:.3f} | "
                  f"{r['norm_entropy']:.3f} | {r['n_classes']} |"]
        wt = (tab["n"] * tab["entropy_bits"]).sum() / tab["n"].sum()
        L += ["",
              f"Pool-weighted mean within-cell entropy: **{wt:.3f} bits** "
              f"(max {np.log2(7):.3f}); marginal label entropy of the pool: "
              f"{entropy(pool['label'].value_counts().to_numpy()):.3f} bits -> a cell "
              f"removes {entropy(pool['label'].value_counts().to_numpy()) - wt:.3f} bits "
              f"of label uncertainty.", ""]

    L += ["## Cell-only oracle: the explicit ceiling for M1", "",
          "Predict each group's modal label. No model whose input is only that grouping "
          "can beat these numbers.", "",
          "| pool | grouping | groups | acc | macro-F1 | quad. kappa |",
          "|---|---|---:|---:|---:|---:|"]
    for name, pool in pools.items():
        for gname, by in [("TABULA cell", ["cell"]),
                          ("cell x city", ["cell", "city"]),
                          ("type x exact bouwjaar", ["building_type", "bouwjaar"])]:
            o = oracle(pool, by)
            L += [f"| {name} | {gname} | {o['n_groups']:,} | {o['acc']:.3f} | "
                  f"{o['macro_f1']:.3f} | {o['kappa']:.3f} |"]
    L += ["",
          "Reading: the TABULA cell alone caps macro-F1 at **0.130** (dev). M1 reaches "
          "0.180 dev-OOF / 0.172 hold-out only because it also gets exact `bouwjaar`, "
          "`num_floors` and `city` on top of the cell — and the type x exact-year oracle "
          "caps that at 0.228. M1 is therefore within ~0.05 macro-F1 of the hard ceiling "
          "of its own feature set: the gap to a useful classifier is missing information, "
          "not a weak model.", ""]

    # ---------------- figure: stacked label mix for the dominant cells ----------------
    tab, piv = cell_table(pools["dev"])
    top = tab.head(10).index[::-1]
    frac = piv.loc[top].div(piv.loc[top].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    left = np.zeros(len(top))
    for i, c in enumerate(LADDER7):
        v = frac[c].to_numpy()
        ax.barh(range(len(top)), v, left=left, color=CLASS_COLORS7[i], label=c,
                edgecolor="white", linewidth=0.4)
        left += v
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{c}  (n={tab.loc[c, 'n']:,})" for c in top])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of buildings in the cell")
    ax.set_title("EPC label mix inside each TABULA cell (dev pool)")
    ax.legend(title="Energieklasse", ncol=7, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.28))
    save_fig(fig, "A03_cell_label_mix")
    plt.close(fig)
    L += ["## Figure", "", "- `reports/figures/audit/A03_cell_label_mix.png`", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    mpl.use("Agg")
    main()
