"""Q3: does feeding SVI-EXTRACTED geometry into the decomposed pipeline help,
and can it beat end-to-end (M2, macro-F1 0.21)?

For each SVI-visible geometry feature (b3_h_max, b3_volume_lod22, shared_ratio):
  L1  extraction: train a RegHead on frozen DINOv2 embeddings -> predict it, OOF R2.
  L2  downstream: feed the PREDICTED values into the energy LightGBM and compare
      - base (S_full)
      - + SVI-predicted geom   <- what the vision pipeline actually gets
      - + GT geom              <- perfect-extraction ceiling (0.226 from the log)

Same dev pool / folds / protocol as svi_compactheid.py. Reuses its RegHead + loaders.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.stage2.features import REPO_ROOT, build_master_table
from src.stage2.metrics import evaluate
from src.stage2.train_eval import FIXED_PARAMS
from src.stage2.svi_compactheid import EMB_COLS, train_head_oof

import torch
from sklearn.metrics import mean_absolute_error, r2_score

CITIES = ["amsterdam", "rotterdam", "utrecht", "delft"]
GEOM = ["b3_h_max", "b3_volume_lod22", "shared_ratio"]


def load_geom() -> pd.DataFrame:
    frames = []
    for c in CITIES:
        p = REPO_ROOT / "data" / "processed" / c / "bag_3dbag_ep_joined.parquet"
        frames.append(pd.read_parquet(
            p, columns=["pand_id", "b3_h_max", "b3_volume_lod22",
                        "b3_opp_buitenmuur", "b3_opp_scheidingsmuur"]))
    g = pd.concat(frames, ignore_index=True)
    g["pand_id"] = g["pand_id"].astype(str)
    denom = g["b3_opp_buitenmuur"] + g["b3_opp_scheidingsmuur"]
    g["shared_ratio"] = np.where(denom > 0, g["b3_opp_scheidingsmuur"] / denom, np.nan)
    return g.drop_duplicates("pand_id").set_index("pand_id")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_master_table(); m["pand_id"] = m["pand_id"].astype(str)
    emb = pd.read_parquet(REPO_ROOT / "reports" / "stage3" / "embeddings_dev.parquet")
    emb["pand_id"] = emb["pand_id"].astype(str)
    m = m.merge(emb, on="pand_id", how="inner")
    g = load_geom()
    for col in GEOM:
        m[col] = m["pand_id"].map(g[col])
    m = m.dropna(subset=GEOM).reset_index(drop=True)

    folds = m["fold"].to_numpy()
    X = m[EMB_COLS].to_numpy(dtype=np.float32)
    y = m["energy_class"]

    # --- L1: extract each geometry feature from SVI ---
    print(f"buildings with embedding + geometry: {len(m)}\n[L1 extraction from SVI]")
    for col in GEOM:
        yv = m[col].to_numpy(dtype=np.float32)
        pred = train_head_oof(X, yv, folds, device)
        m["svi_" + col] = pred
        print(f"  SVI -> {col:20s} R2={r2_score(yv, pred):.3f} "
              f"MAE={mean_absolute_error(yv, pred):.3f}")

    # --- L2: downstream energy classification ---
    cats = ["building_type", "city"]
    base = ["building_type", "bouwjaar", "u_wall", "u_roof", "u_floor", "u_window",
            "num_floors", "city"]

    def oof(cols):
        Xd = m[cols].copy()
        for c in cats:
            if c in cols:
                Xd[c] = Xd[c].astype("category")
        pred = np.empty(len(m), dtype=object)
        for f in range(5):
            tr, va = folds != f, folds == f
            cl = LGBMClassifier(**FIXED_PARAMS)
            cl.fit(Xd[tr], y[tr], categorical_feature=[c for c in cats if c in cols])
            pred[va] = cl.predict(Xd[va])
        return evaluate(pd.DataFrame({"true": y.values, "pred": pred}), with_ci=False)

    svi_cols = ["svi_" + c for c in GEOM]
    print(f"\n[L2 downstream energy] (n={len(m)})  end-to-end M2 ref: macro-F1 0.21")
    for name, cols in [
        ("base (S_full)", base),
        ("+ SVI-predicted geom", base + svi_cols),
        ("+ GT geom (ceiling)", base + GEOM),
    ]:
        r = oof(cols)
        print(f"  {name:24s} macroF1={r['macro_f1']:.4f} "
              f"kappa={r['quadratic_kappa']:.4f} acc={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
