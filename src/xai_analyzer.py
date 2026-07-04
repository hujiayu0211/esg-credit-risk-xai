"""XGBoostXAIAnalyzer: multi-layer SHAP explainability for tree models.

Four analysis layers, mirroring how model behaviour is typically reported
in applied ML / finance research:

  1. Global feature importance   -> beeswarm + mean(|SHAP|) bar chart
  2. Dependence analysis         -> SHAP dependence plots with LOWESS smoothing
  3. Interaction effects         -> SHAP interaction-value matrix (heatmap)
  4. Individual explanations     -> per-observation force / waterfall plots

Usage:
    analyzer = XGBoostXAIAnalyzer(fitted_pipeline, X_test, output_dir="figures")
    analyzer.global_importance()
    analyzer.dependence(["esg_score", "leverage"])
    analyzer.interaction_matrix()
    analyzer.explain_instance(0)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class XGBoostXAIAnalyzer:
    """Encapsulates the full SHAP explainability workflow for a fitted
    sklearn Pipeline whose final step is an XGBRegressor."""

    def __init__(self, pipeline, X: pd.DataFrame, output_dir: str = "figures",
                 interaction_sample: int = 300, seed: int = 42):
        self.pipeline = pipeline
        self.feature_names = list(X.columns)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Apply the pipeline's preprocessing so SHAP sees what the model sees
        imputer = pipeline.named_steps["imputer"]
        self.model = pipeline.named_steps["xgb"]
        self.X = pd.DataFrame(imputer.transform(X), columns=self.feature_names,
                              index=X.index)

        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X)

        rng = np.random.default_rng(seed)
        n_sub = min(interaction_sample, len(self.X))
        self._sub_idx = rng.choice(len(self.X), n_sub, replace=False)
        self._interaction_values = None  # computed lazily (expensive)

    # ------------------------------------------------------------------
    # Layer 1: global importance
    # ------------------------------------------------------------------
    def global_importance(self, top_n: int = 18) -> pd.DataFrame:
        """Beeswarm summary plot + mean(|SHAP|) bar chart.
        Returns the importance table sorted descending."""
        # Beeswarm
        plt.figure()
        shap.summary_plot(self.shap_values, self.X, show=False, max_display=top_n)
        plt.title("SHAP Summary: Distribution of Feature Impacts on Credit Risk")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_summary_beeswarm.png", bbox_inches="tight")
        plt.close()

        # Mean |SHAP| bar
        importance = (
            pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(self.shap_values).mean(axis=0),
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        top = importance.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2d6a4f" if f.endswith("_score") else "#40587c"
                  for f in top["feature"]]
        ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
        ax.set_xlabel("mean(|SHAP value|)  (impact on predicted credit risk)")
        ax.set_title("Global Feature Importance (ESG features in green)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_importance_bar.png", bbox_inches="tight")
        plt.close()
        return importance

    # ------------------------------------------------------------------
    # Layer 2: dependence with LOWESS smoothing
    # ------------------------------------------------------------------
    def dependence(self, features: list[str], lowess_frac: float = 0.3) -> None:
        """Scatter of SHAP value vs feature value, coloured by the feature
        with the strongest automatic interaction, overlaid with a LOWESS
        trend line."""
        for feat in features:
            j = self.feature_names.index(feat)
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(feat, self.shap_values, self.X, ax=ax, show=False,
                                 alpha=0.35, dot_size=10)
            smoothed = lowess(self.shap_values[:, j], self.X[feat].values,
                              frac=lowess_frac, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#c1121f", lw=2.5,
                    label=f"LOWESS (frac={lowess_frac})")
            ax.legend(loc="best", frameon=False)
            ax.set_title(f"SHAP Dependence: {feat}", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/shap_dependence_{feat}.png",
                        bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------------
    # Layer 3: interaction effects
    # ------------------------------------------------------------------
    def interaction_matrix(self, top_n: int = 10) -> pd.DataFrame:
        """Mean(|SHAP interaction value|) matrix for the top_n features,
        rendered as a heatmap. Computed on a random subsample because
        interaction values scale O(n * p^2)."""
        if self._interaction_values is None:
            self._interaction_values = self.explainer.shap_interaction_values(
                self.X.iloc[self._sub_idx]
            )
        inter = np.abs(self._interaction_values).mean(axis=0)
        np.fill_diagonal(inter, 0)  # suppress main effects to expose interactions

        order = np.argsort(-np.abs(self.shap_values).mean(axis=0))[:top_n]
        labels = [self.feature_names[i] for i in order]
        sub = inter[np.ix_(order, order)]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sub, cmap="Greens")
        ax.set_xticks(range(top_n), labels, rotation=45, ha="right")
        ax.set_yticks(range(top_n), labels)
        for i in range(top_n):
            for k in range(top_n):
                ax.text(k, i, f"{sub[i, k]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if sub[i, k] > sub.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean(|SHAP interaction value|)")
        ax.set_title("Pairwise SHAP Interaction Effects (off-diagonal)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_interaction_heatmap.png",
                    bbox_inches="tight")
        plt.close()
        return pd.DataFrame(sub, index=labels, columns=labels)

    def interaction_dependence(self, feat: str, interact_with: str) -> None:
        """Dependence plot of feat explicitly coloured by interact_with."""
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat, self.shap_values, self.X,
                             interaction_index=interact_with, ax=ax, show=False,
                             alpha=0.4, dot_size=10)
        ax.set_title(f"Interaction: {feat} x {interact_with}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_interaction_{feat}_x_{interact_with}.png",
                    bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Layer 4: individual-level explanations
    # ------------------------------------------------------------------
    def explain_instance(self, i: int, tag: str | None = None) -> None:
        """Force plot + waterfall plot for observation i (positional)."""
        tag = tag or str(i)
        # Force plot
        shap.force_plot(self.explainer.expected_value, self.shap_values[i],
                        self.X.iloc[i].round(3), matplotlib=True, show=False,
                        figsize=(18, 3.2))
        plt.title(f"Force Plot: firm-year observation #{tag}", fontsize=11)
        plt.savefig(f"{self.output_dir}/shap_force_obs_{tag}.png",
                    bbox_inches="tight")
        plt.close()

        # Waterfall plot
        expl = shap.Explanation(values=self.shap_values[i],
                                base_values=self.explainer.expected_value,
                                data=self.X.iloc[i].values,
                                feature_names=self.feature_names)
        plt.figure()
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.title(f"Waterfall: observation #{tag}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_waterfall_obs_{tag}.png",
                    bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    def run_all(self, dependence_features: list[str],
                instance_ids: list[int]) -> pd.DataFrame:
        """Run all four layers and return the global importance table."""
        importance = self.global_importance()
        self.dependence(dependence_features)
        self.interaction_matrix()
        for i in instance_ids:
            self.explain_instance(i)
        return importance
