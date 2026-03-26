import sys
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# ============================================================
# Utils
# ============================================================

def check_and_create_path(path: Path):
    path_way = path.parent if path.suffix != "" else path
    path_way.mkdir(parents=True, exist_ok=True)


def save_object(obj, filename: str, path: Path):
    check_and_create_path(path)
    with open(path / filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def graph_id2_num_zone(df, scale, graph_construct):
    df = df.copy()

    if scale == 'departement' or "zonemeteo" not in graph_construct:
        df['num_zone'] = df['graph_id'].values
        return df

    df['num_zone'] = df['graph_id'].values

    mask_dep6 = df['departement'] == 6
    if True in np.unique(mask_dep6):
        df_dep6 = df[mask_dep6]
        graph_ids_6 = np.sort(df_dep6['graph_id'].unique())
        num_zone_6 = [65, 62, 64, 61, 66, 67, 63]

        if len(graph_ids_6) != len(num_zone_6):
            print(f"Removing first id {graph_ids_6[0]}")
            graph_ids_6 = graph_ids_6[1:]

        dico_6 = {gi: num_zone_6[i] for i, gi in enumerate(graph_ids_6)}
        df.loc[mask_dep6, 'num_zone'] = df.loc[mask_dep6, 'graph_id'].map(dico_6)

    mask_other = df['departement'] != 6
    df_other = df[mask_other]
    graph_ids_other = np.sort(df_other['graph_id'].unique())
    dico_other = {gi: gi for gi in graph_ids_other}
    df.loc[mask_other, 'num_zone'] = df.loc[mask_other, 'graph_id'].map(dico_other)

    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    extra_exclude: Optional[List[str]] = None
) -> List[str]:
    exclude = {
        "date", "departement", "graph_id", "id", "num_zone", "season",
        "latitude", "longitude", "weight"
    }
    if target_col is not None:
        exclude.add(target_col)
    if extra_exclude is not None:
        exclude.update(extra_exclude)

    features = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
    return features


def make_quantile_grid(
    s: pd.Series,
    n_points: int = 9,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.array([])
    vals = np.quantile(s.values, np.linspace(q_low, q_high, n_points))
    vals = np.unique(vals)
    return vals


# ============================================================
# Prediction
# ============================================================

def predict_class(model, df: pd.DataFrame) -> np.ndarray:
    """
    Utilise explicitement la méthode fournie par l'utilisateur :
        model.predict(df, prediction_type="Class")
    et retourne un vecteur numpy shape (n,).
    """
    pred = model.predict(df=df, graph=None, return_y=False, prediction_type="Class")

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    elif isinstance(pred, pd.Series):
        pred = pred.values
    elif isinstance(pred, pd.DataFrame):
        pred = pred.values

    pred = np.asarray(pred).reshape(-1)
    return pred.astype(float)

import numpy as np
import pandas as pd
import torch


def predict_raw_formula_val(model, df: pd.DataFrame) -> np.ndarray:
    """
    Utilise:
        model.predict(df=df, prediction_type='RawFormulaVal')
    et retourne un array numpy de shape:
      - (n, 1) si sortie scalaire
      - (n, k) si sortie multi-dimensionnelle
    """

    pred = model.predict(
        df=df,
        graph=None,
        return_y=False,
        prediction_type="RawFormulaVal"
    )

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    elif isinstance(pred, pd.Series):
        pred = pred.values
    elif isinstance(pred, pd.DataFrame):
        pred = pred.values

    pred = np.asarray(pred)

    # Cas fréquents :
    # (n,)       -> (n,1)
    # (n,1)      -> (n,1)
    # (n,1,1)    -> (n,1)
    # (n,1,k)    -> (n,k)
    # (n,k,1)    -> (n,k)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    elif pred.ndim >= 2:
        # on conserve la dimension batch (axe 0)
        # et on aplatit toutes les autres dimensions
        pred = pred.reshape(pred.shape[0], -1)

    else:
        raise ValueError(f"Shape de sortie inattendue pour RawFormulaVal: {pred.shape}")

    return pred.astype(float)

# ============================================================
# Intervention analysis
# ============================================================

def classify_curve_sign(curve: np.ndarray, tol: float = 1e-12) -> str:
    if len(curve) < 2:
        return "undetermined"
    diffs = np.diff(curve)
    if np.all(diffs >= -tol) and np.any(diffs > tol):
        return "positive"
    if np.all(diffs <= tol) and np.any(diffs < -tol):
        return "negative"
    if np.all(np.abs(diffs) <= tol):
        return "flat"
    return "mixed"


def compute_interventional_curve_raw(
    model,
    df_eval: pd.DataFrame,
    df_train: pd.DataFrame,
    feature: str,
    get_local_historical_values_fn,
    local_mode: str = "zone",
    n_grid: int = 9,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_eval_size=None,
    random_state: int = 42,
):
    """
    Estime, pour chaque dimension de sortie brute du modèle:
        mu_j(v) = E[ raw_output | do(x_j = v) ]

    Retourne:
      - curve_df long-format
      - summary_df par output_idx
    """
    dd = df_eval.copy()

    if max_eval_size is not None and len(dd) > max_eval_size:
        dd = dd.sample(max_eval_size, random_state=random_state).copy()

    grid = get_local_historical_values_fn(
        df_train=df_train,
        df_eval=dd,
        feature=feature,
        local_mode=local_mode,
        n_grid=n_grid,
        q_low=q_low,
        q_high=q_high,
    )

    if len(grid) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for v in grid:
        dmod = dd.copy()
        dmod[feature] = v

        pred = predict_raw_formula_val(model, dmod)  # shape (n, k)

        for output_idx in range(pred.shape[1]):
            vals = pred[:, output_idx]

            rows.append({
                "feature": feature,
                "value": float(v),
                "output_idx": int(output_idx),
                "pred_mean": float(np.mean(vals)),
                "pred_std": float(np.std(vals)),
                "pred_median": float(np.median(vals)),
                "pred_q10": float(np.quantile(vals, 0.10)),
                "pred_q25": float(np.quantile(vals, 0.25)),
                "pred_q75": float(np.quantile(vals, 0.75)),
                "pred_q90": float(np.quantile(vals, 0.90)),
                "n_eval": len(dmod),
            })

    curve_df = pd.DataFrame(rows)

    summary_rows = []
    for output_idx, sdf in curve_df.groupby("output_idx"):
        mean_curve = sdf.sort_values("value")["pred_mean"].values
        summary_rows.append({
            "feature": feature,
            "output_idx": int(output_idx),
            "n_eval": len(dd),
            "n_grid": len(grid),
            "effect_std": float(np.std(mean_curve)),
            "effect_range": float(np.max(mean_curve) - np.min(mean_curve)),
            "curve_min": float(np.min(mean_curve)),
            "curve_max": float(np.max(mean_curve)),
            "sign": classify_curve_sign(mean_curve),
        })

    summary_df = pd.DataFrame(summary_rows)
    return curve_df, summary_df

def make_quantile_grid(
    s: pd.Series,
    n_points: int = 9,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.array([])
    vals = np.quantile(s.values, np.linspace(q_low, q_high, n_points))
    vals = np.unique(vals)
    return vals


def get_local_historical_values(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature: str,
    local_mode: str = "zone",
    n_grid: int = 9,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    hist = df_train.copy()

    if local_mode == "global":
        pass

    elif local_mode == "departement":
        deps = df_eval["departement"].dropna().unique()
        if len(deps) == 1:
            hist = hist[hist["departement"] == deps[0]]

    elif local_mode == "zone":
        zones = df_eval["num_zone"].dropna().unique()
        if len(zones) == 1:
            hist = hist[hist["num_zone"] == zones[0]]
            
    elif local_mode == "saison":
        saisons = df_eval["saison"].dropna().unique()
        if len(saisons) == 1:
            hist = hist[hist["saison"] == saisons[0]]

    else:
        raise ValueError(f"local_mode inconnu: {local_mode}")

    vals = make_quantile_grid(hist[feature], n_points=n_grid, q_low=q_low, q_high=q_high)

    if len(vals) == 0:
        vals = make_quantile_grid(df_train[feature], n_points=n_grid, q_low=q_low, q_high=q_high)

    return vals

import matplotlib.pyplot as plt
from pathlib import Path


def check_and_create_path(path: Path):
    path_way = path.parent if path.suffix != "" else path
    path_way.mkdir(parents=True, exist_ok=True)


def plot_interventional_curve_raw(
    curve_df: pd.DataFrame,
    title: str,
    save_path: Path
):
    """
    Trace une figure par output_idx.
    Courbe centrale = moyenne
    Bandes = [q25, q75] et [q10, q90]
    """
    if len(curve_df) == 0:
        return

    check_and_create_path(save_path)

    for output_idx, sdf in curve_df.groupby("output_idx"):
        sdf = sdf.sort_values("value")

        x = sdf["value"].values
        y = sdf["pred_mean"].values
        q10 = sdf["pred_q10"].values
        q25 = sdf["pred_q25"].values
        q75 = sdf["pred_q75"].values
        q90 = sdf["pred_q90"].values

        plt.figure(figsize=(7, 5))
        plt.plot(x, y, marker="o", label="E[raw output]")
        plt.fill_between(x, q25, q75, alpha=0.30, label="q25-q75")
        plt.fill_between(x, q10, q90, alpha=0.15, label="q10-q90")
        plt.xlabel("Intervened feature value")
        plt.ylabel(f"Raw output {output_idx}")
        plt.title(f"{title} | output_{output_idx}")
        plt.legend()
        plt.tight_layout()

        out = save_path.with_name(f"{save_path.stem}_output{output_idx}{save_path.suffix}")
        plt.savefig(out, dpi=150)
        plt.close()

def run_analysis_for_group_raw(
    model,
    df_eval: pd.DataFrame,
    df_train: pd.DataFrame,
    features,
    output_dir: Path,
    group_name: str,
    get_local_historical_values_fn,
    local_mode: str = "zone",
    n_grid: int = 9,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_eval_size=None,
    random_state: int = 42,
):
    summary_list = []

    curve_dir = output_dir / group_name / "curves"
    table_dir = output_dir / group_name / "tables"
    check_and_create_path(curve_dir)
    check_and_create_path(table_dir)

    for feature in features:
        try:
            curve_df, summary_df = compute_interventional_curve_raw(
                model=model,
                df_eval=df_eval,
                df_train=df_train,
                feature=feature,
                get_local_historical_values_fn=get_local_historical_values_fn,
                local_mode=local_mode,
                n_grid=n_grid,
                q_low=q_low,
                q_high=q_high,
                max_eval_size=max_eval_size,
                random_state=random_state,
            )

            if len(curve_df) > 0:
                curve_df.to_csv(table_dir / f"{feature}.csv", index=False)

                plot_interventional_curve_raw(
                    curve_df=curve_df,
                    title=f"{group_name} | {feature}",
                    save_path=curve_dir / f"{feature}.png"
                )

            if len(summary_df) > 0:
                summary_list.append(summary_df)

        except Exception as e:
            print(f"[ERROR] group={group_name}, feature={feature}: {e}")

    if len(summary_list) > 0:
        summary_all = pd.concat(summary_list, axis=0, ignore_index=True)
        summary_all = summary_all.sort_values(
            by=["effect_std", "effect_range"],
            ascending=False
        ).reset_index(drop=True)
    else:
        summary_all = pd.DataFrame()

    summary_all.to_csv(output_dir / group_name / "summary.csv", index=False)
    return summary_all


# ============================================================
# Main
# ============================================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
prediction_path = current_dir.parent

if str(prediction_path) not in sys.path:
    sys.path.append(str(prediction_path))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

model_name = 'GRU_search_full_10_0_all_one_nbsinister_regression_ccllt-id{node}-nclusters{30}'
root_path = f"/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_06_01_25_78/check_z-score/full_all_3_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node/{model_name}"
model_path = f"{root_path}/{model_name}.pkl"

print(f"Chargement du modèle: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

device = "cpu"
model.device = device
if hasattr(model, "model") and model.model is not None:
    model.model.to(device)

# =====================
# Données
# =====================
df_train = model.df_train.copy(deep=True)
df_test = model.df_test.copy(deep=True)

target_name = model.target_name

df_train = graph_id2_num_zone(df_train, scale=3, graph_construct='risk-size-zonemeteo-degree-a3-r5-t0.3')
df_test = graph_id2_num_zone(df_test, scale=3, graph_construct='risk-size-zonemeteo-degree-a3-r5-t0.3')

features = model.features_name

print(f"Nombre de features numériques retenues: {len(features)}")

# =====================
# Répertoire de sortie
# =====================
output_dir = Path(root_path) / "interventional_analysis_class"
check_and_create_path(output_dir)

# Sauvegarde d'un méta fichier
meta = {
    "model_name": model_name,
    "target_name": target_name,
    "n_train": len(df_train),
    "n_test": len(df_test),
    "n_features": len(features),
    "features": features,
}
save_object(meta, "meta.pkl", output_dir)

# ============================================================
# 1) Analyse globale
# ============================================================
print("Analyse globale...")
summary_global = run_analysis_for_group_raw(
    model=model,
    df_eval=df_test,
    df_train=df_train,
    features=features,
    output_dir=output_dir,
    group_name="global",
    get_local_historical_values_fn=get_local_historical_values,
    local_mode="zone",
    n_grid=9,
    q_low=0.05,
    q_high=0.95,
    max_eval_size=None,
    random_state=42
)

# ============================================================
# 2) Analyse par département
# ============================================================
# ============================================================
# 2) Analyse par département
# ============================================================
if "departement" in df_test.columns:
    for dept in sorted(df_test["departement"].dropna().unique()):
        print(f"Analyse département {dept}...")
        df_eval_dept = df_test[df_test["departement"] == dept].copy()

        run_analysis_for_group_raw(
            model=model,
            df_eval=df_eval_dept,
            df_train=df_train,
            features=features,
            output_dir=output_dir,
            group_name=f"departement_{dept}",
            get_local_historical_values_fn=get_local_historical_values,
            local_mode="departement",
            n_grid=9,
            q_low=0.05,
            q_high=0.95,
            max_eval_size=None,
            random_state=42
        )

# ============================================================
# 3) Analyse par zone
# ============================================================
if "num_zone" in df_test.columns:
    for zone in sorted(df_test["num_zone"].dropna().unique()):
        print(f"Analyse zone {zone}...")
        df_eval_zone = df_test[df_test["num_zone"] == zone].copy()

        run_analysis_for_group_raw(
            model=model,
            df_eval=df_eval_zone,
            df_train=df_train,
            features=features,
            output_dir=output_dir,
            group_name=f"zone_{zone}",
            get_local_historical_values_fn=get_local_historical_values,
            local_mode="zone",
            n_grid=9,
            q_low=0.05,
            q_high=0.95,
            max_eval_size=None,
            random_state=42
        )

# ============================================================
# 4) Analyse par saison
# ============================================================
if "saison" in df_test.columns:
    for season in sorted(df_test["saison"].dropna().unique()):
        print(f"Analyse saison {season}...")
        df_eval_season = df_test[df_test["saison"] == season].copy()

        run_analysis_for_group_raw(
            model=model,
            df_eval=df_eval_season,
            df_train=df_train,
            features=features,
            output_dir=output_dir,
            group_name=f"saison_{season}",
            get_local_historical_values_fn=get_local_historical_values,
            local_mode="saison",
            n_grid=9,
            q_low=0.05,
            q_high=0.95,
            max_eval_size=None,
            random_state=42
        )

print(f"Analyse terminée. Résultats dans : {output_dir}")