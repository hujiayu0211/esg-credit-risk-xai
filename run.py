"""End-to-end reproduction script.

    python run.py                       # uses data/sample_synthetic.csv
    python run.py --data my_panel.csv   # uses a real CSMAR extract

Regenerates the synthetic sample if it does not exist, tunes the XGBoost
pipeline (RandomizedSearchCV, 28 iterations x 5-fold CV), evaluates on a
held-out test set, and produces all SHAP figures in figures/.
"""

import argparse
import json
import os
import time

import yaml

from src.data_generator import generate_panel
from src.data_prep import load_dataset, split
from src.model import tune, evaluate
from src.xai_analyzer import XGBoostXAIAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to input CSV")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = args.data or cfg["data"]["path"]
    if not os.path.exists(data_path):
        print(f"[1/4] {data_path} not found -> generating synthetic sample")
        panel = generate_panel(**cfg["data"]["synthetic"])
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        panel.to_csv(data_path, index=False)
    else:
        print(f"[1/4] Loading {data_path}")

    X, y, _ = load_dataset(data_path, winsor=cfg["data"]["winsorize"])
    X_train, X_test, y_train, y_test = split(
        X, y, test_size=cfg["split"]["test_size"], seed=cfg["seed"])
    print(f"      {len(X_train)} train / {len(X_test)} test observations, "
          f"{X.shape[1]} features")

    print(f"[2/4] Tuning: RandomizedSearchCV "
          f"({cfg['tuning']['n_iter']} iterations x {cfg['tuning']['cv']}-fold CV)")
    t0 = time.time()
    search = tune(X_train, y_train, n_iter=cfg["tuning"]["n_iter"],
                  cv=cfg["tuning"]["cv"], seed=cfg["seed"])
    print(f"      done in {time.time() - t0:.0f}s | best CV RMSE: "
          f"{-search.best_score_:.3f}")

    print("[3/4] Evaluating on held-out test set")
    metrics = evaluate(search.best_estimator_, X_test, y_test)
    print(f"      RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
          f"R2={metrics['r2']:.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"test": metrics,
                   "cv_best_rmse": -search.best_score_,
                   "best_params": search.best_params_}, f, indent=2, default=str)

    print("[4/4] SHAP explainability analysis -> figures/")
    analyzer = XGBoostXAIAnalyzer(search.best_estimator_, X_test,
                                  output_dir=cfg["xai"]["output_dir"],
                                  seed=cfg["seed"])
    importance = analyzer.run_all(
        dependence_features=cfg["xai"]["dependence_features"],
        instance_ids=cfg["xai"]["instance_ids"])
    analyzer.interaction_dependence("esg_score", "leverage")
    importance.to_csv("outputs/feature_importance.csv", index=False)

    print("\nTop 5 features by mean(|SHAP|):")
    print(importance.head(5).to_string(index=False))
    print("\nModel + SHAP done. Run the evaluation suite for the ESG study:")
    print("  python -m src.evaluation.ablation")
    print("  python -m src.evaluation.distress_classification")
    print("  python -m src.evaluation.robustness")
    print("\nAll figures in figures/, metrics in outputs/")


if __name__ == "__main__":
    main()
