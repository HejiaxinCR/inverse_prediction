from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .data_bounds import append_reference_features, compute_feature_bounds, rebuild_training_frame
    from .model import (
        BASE_FEATURE_NAMES,
        FEATURE_NAMES,
        REFERENCE_FEATURE_NAMES,
        TARGET_NAMES,
        MLPSurrogate,
    )
except ImportError:
    from data_bounds import append_reference_features, compute_feature_bounds, rebuild_training_frame
    from model import (
        BASE_FEATURE_NAMES,
        FEATURE_NAMES,
        REFERENCE_FEATURE_NAMES,
        TARGET_NAMES,
        MLPSurrogate,
    )


@dataclass
class InverseOptimizationResult:
    x_initial_raw: np.ndarray
    x_result_raw: np.ndarray
    initial_pred_raw: np.ndarray
    final_pred_raw: np.ndarray
    target_raw: np.ndarray
    adjustable_names: list[str]
    feature_table: pd.DataFrame
    history: pd.DataFrame
    touched_bounds: pd.DataFrame
    fixed_out_of_bounds: pd.DataFrame


class InverseOptimizer:
    REQUIRED_ARTIFACT_FILES = (
        "normalizer_x.pkl",
        "normalizer_y.pkl",
        "mlp_surrogate_state_dict.pt",
    )
    OUTPUT_SPECS = [
        ("sa", "target_sa", "w_sa", "keep_sa_weight"),
        ("ua", "target_ua", "w_ua", "keep_ua_weight"),
        ("lambda_airin", "target_lambda_airin", "w_lambda_airin", "keep_lambda_airin_weight"),
    ]

    def __init__(
        self,
        artifact_dir=Path("module_value"),
        device=None,
        train_df=None,
    ):
        self.artifact_dir = self._resolve_artifact_dir(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.normalizer_x = self._load_pickle(self.artifact_dir / "normalizer_x.pkl")
        self.normalizer_y = self._load_pickle(self.artifact_dir / "normalizer_y.pkl")

        self.model = MLPSurrogate(
            input_dim=len(FEATURE_NAMES),
            output_dim=len(TARGET_NAMES),
        ).to(self.device)
        state_dict = torch.load(
            self.artifact_dir / "mlp_surrogate_state_dict.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if train_df is None:
            train_df = rebuild_training_frame()
        self.train_df = train_df
        self.base_feature_names = list(BASE_FEATURE_NAMES)
        self.reference_feature_names = list(REFERENCE_FEATURE_NAMES)
        self.base_idx = [FEATURE_NAMES.index(name) for name in self.base_feature_names]
        self.reference_idx = [FEATURE_NAMES.index(name) for name in self.reference_feature_names]
        self.x_min_raw, self.x_max_raw = compute_feature_bounds(train_df)
        self.x_min_norm = self._transform_x(self.x_min_raw.to_numpy(dtype=np.float32))
        self.x_max_norm = self._transform_x(self.x_max_raw.to_numpy(dtype=np.float32))

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def _resolve_artifact_dir(cls, artifact_dir):
        artifact_dir = Path(artifact_dir)
        script_dir = Path(__file__).resolve().parent
        candidates = []

        if artifact_dir.is_absolute():
            candidates.append(artifact_dir)
        else:
            candidates.append(Path.cwd() / artifact_dir)
            if artifact_dir.parts and artifact_dir.parts[0] == script_dir.name:
                candidates.append(script_dir.joinpath(*artifact_dir.parts[1:]))
            candidates.append(script_dir / artifact_dir)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_candidates.append(resolved)

        for candidate in unique_candidates:
            if all((candidate / name).is_file() for name in cls.REQUIRED_ARTIFACT_FILES):
                return candidate

        checked = "\n  ".join(str(candidate) for candidate in unique_candidates)
        required = ", ".join(cls.REQUIRED_ARTIFACT_FILES)
        raise FileNotFoundError(
            f"Could not find model artifacts ({required}). Checked:\n  {checked}"
        )

    def _transform_x(self, x_raw):
        x_raw = np.asarray(x_raw, dtype=np.float32).reshape(1, -1)
        return self.normalizer_x.transform(x_raw)[0].astype(np.float32)

    def _inverse_x(self, x_norm):
        x_norm = np.asarray(x_norm, dtype=np.float32).reshape(1, -1)
        return self.normalizer_x.inverse_transform(x_norm)[0].astype(np.float32)

    def _transform_y(self, y_raw):
        y_raw = np.asarray(y_raw, dtype=np.float32).reshape(1, -1)
        return self.normalizer_y.transform(y_raw)[0].astype(np.float32)

    def _inverse_y(self, y_norm):
        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(1, -1)
        return self.normalizer_y.inverse_transform(y_norm)[0].astype(np.float32)

    def _ensure_full_feature_array(self, x_raw):
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if x_raw.shape == (len(FEATURE_NAMES),):
            return x_raw
        if x_raw.shape != (len(BASE_FEATURE_NAMES),):
            raise ValueError(
                f"x_raw must have shape ({len(BASE_FEATURE_NAMES)},) or ({len(FEATURE_NAMES)},), "
                f"got {x_raw.shape}"
            )
        frame = append_reference_features(
            pd.DataFrame([x_raw], columns=BASE_FEATURE_NAMES)
        )
        return frame.loc[0, FEATURE_NAMES].to_numpy(dtype=np.float32)

    def _refresh_reference_features(self, x_norm_tensor):
        full_raw = self._inverse_x(x_norm_tensor.detach().cpu().numpy())
        base_raw = full_raw[self.base_idx]
        refreshed_raw = self._ensure_full_feature_array(base_raw)
        refreshed_norm = self._transform_x(refreshed_raw)
        x_norm_tensor[self.reference_idx] = torch.tensor(
            refreshed_norm[self.reference_idx],
            dtype=torch.float32,
            device=self.device,
        )
        return refreshed_raw

    def predict_raw(self, x_raw):
        x_raw = self._ensure_full_feature_array(x_raw)
        x_norm = torch.tensor(
            self._transform_x(x_raw),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self.model(x_norm).cpu().numpy()[0]
        return self._inverse_y(pred_norm)

    def optimize(
        self,
        x0_raw,
        target_sa=None,
        target_ua=None,
        target_lambda_airin=None,
        adjustable_names=("I", "STC_ano", "RHano_in"),
        lr=0.01,
        n_steps=500,
        reg_weight=0.01,
        w_sa=1.0,
        w_ua=1.0,
        w_lambda_airin=1.0,
        keep_sa_weight=0.0,
        keep_ua_weight=0.0,
        keep_lambda_airin_weight=0.0,
        print_every=50,
    ):
        requested_targets = {
            "sa": target_sa,
            "ua": target_ua,
            "lambda_airin": target_lambda_airin,
        }
        if all(value is None for value in requested_targets.values()):
            raise ValueError(
                "Provide at least one of target_sa, target_ua, or target_lambda_airin."
            )

        x0_raw = self._ensure_full_feature_array(x0_raw)

        unknown = sorted(set(adjustable_names) - set(BASE_FEATURE_NAMES))
        if unknown:
            raise ValueError(f"Unknown adjustable feature names: {unknown}")

        adjustable_idx = [FEATURE_NAMES.index(name) for name in adjustable_names]
        fixed_base_idx = [idx for idx in self.base_idx if idx not in adjustable_idx]

        x0_norm_np = self._transform_x(x0_raw)
        x_norm = torch.tensor(
            x0_norm_np,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        x_norm_orig = x_norm.detach().clone()

        initial_pred_raw = self.predict_raw(x0_raw)
        target_values = {
            "sa": target_sa,
            "ua": target_ua,
            "lambda_airin": target_lambda_airin,
        }
        loss_weights = {
            "sa": w_sa,
            "ua": w_ua,
            "lambda_airin": w_lambda_airin,
        }
        keep_weights = {
            "sa": keep_sa_weight,
            "ua": keep_ua_weight,
            "lambda_airin": keep_lambda_airin_weight,
        }
        target_raw = np.array(
            [
                np.nan if target_values[name] is None else target_values[name]
                for name in TARGET_NAMES
            ],
            dtype=np.float32,
        )
        reference_raw = np.array(
            [
                initial_pred_raw[idx]
                if target_values[name] is None
                else target_values[name]
                for idx, name in enumerate(TARGET_NAMES)
            ],
            dtype=np.float32,
        )
        reference_norm = torch.tensor(
            self._transform_y(reference_raw),
            dtype=torch.float32,
            device=self.device,
        )

        x_min_norm = torch.tensor(self.x_min_norm, dtype=torch.float32, device=self.device)
        x_max_norm = torch.tensor(self.x_max_norm, dtype=torch.float32, device=self.device)
        grad_mask = torch.zeros_like(x_norm)
        grad_mask[adjustable_idx] = 1.0

        optimizer = torch.optim.Adam([x_norm], lr=lr)
        history_rows = []

        for step in range(n_steps):
            with torch.no_grad():
                self._refresh_reference_features(x_norm)

            optimizer.zero_grad()
            pred_norm = self.model(x_norm.unsqueeze(0))[0]

            loss_target = pred_norm.new_tensor(0.0)
            for idx, name in enumerate(TARGET_NAMES):
                if target_values[name] is not None:
                    loss_target = loss_target + loss_weights[name] * (
                        pred_norm[idx] - reference_norm[idx]
                    ) ** 2
                elif keep_weights[name] > 0:
                    loss_target = loss_target + keep_weights[name] * (
                        pred_norm[idx] - reference_norm[idx]
                    ) ** 2

            diff = x_norm - x_norm_orig
            loss_reg = (diff[adjustable_idx] ** 2).sum() * reg_weight
            loss = loss_target + loss_reg
            loss.backward()

            x_norm.grad *= grad_mask
            optimizer.step()

            with torch.no_grad():
                x_norm.clamp_(min=x_min_norm, max=x_max_norm)
                if fixed_base_idx:
                    x_norm[fixed_base_idx] = x_norm_orig[fixed_base_idx]
                self._refresh_reference_features(x_norm)

            pred_raw = self._inverse_y(pred_norm.detach().cpu().numpy())
            history_row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "loss_target": float(loss_target.detach().cpu().item()),
                "loss_reg": float(loss_reg.detach().cpu().item()),
            }
            for idx, name in enumerate(TARGET_NAMES):
                history_row[f"{name}_pred"] = float(pred_raw[idx])
            history_rows.append(history_row)

            if print_every is not None and (step == 0 or (step + 1) % print_every == 0):
                print(
                    f"step {step + 1:04d}: "
                    f"sa={pred_raw[0]:.6f}, ua={pred_raw[1]:.6f}, "
                    f"lambda_airin={pred_raw[2]:.6f}, loss={loss.item():.6f}"
                )

        x_result_norm = x_norm.detach().cpu().numpy()
        x_result_raw = self._inverse_x(x_result_norm)
        final_pred_raw = self.predict_raw(x_result_raw)
        history = pd.DataFrame(history_rows)

        feature_table = self._make_feature_table(x0_raw, x_result_raw, adjustable_names)
        touched_bounds = self._make_touched_bounds(feature_table)
        fixed_out_of_bounds = self._make_fixed_out_of_bounds(feature_table)

        return InverseOptimizationResult(
            x_initial_raw=x0_raw,
            x_result_raw=x_result_raw,
            initial_pred_raw=initial_pred_raw,
            final_pred_raw=final_pred_raw,
            target_raw=target_raw,
            adjustable_names=list(adjustable_names),
            feature_table=feature_table,
            history=history,
            touched_bounds=touched_bounds,
            fixed_out_of_bounds=fixed_out_of_bounds,
        )

    def _make_feature_table(self, x0_raw, x_result_raw, adjustable_names):
        rows = []
        for name in BASE_FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            raw_min = float(self.x_min_raw[name])
            raw_max = float(self.x_max_raw[name])
            initial = float(x0_raw[idx])
            result = float(x_result_raw[idx])
            rows.append(
                {
                    "feature": name,
                    "initial": initial,
                    "optimized": result,
                    "delta": result - initial,
                    "train_min": raw_min,
                    "train_max": raw_max,
                    "adjustable": name in adjustable_names,
                    "at_lower_bound": np.isclose(result, raw_min, rtol=0, atol=1e-6),
                    "at_upper_bound": np.isclose(result, raw_max, rtol=0, atol=1e-6),
                    "initial_out_of_bounds": initial < raw_min or initial > raw_max,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_touched_bounds(feature_table):
        mask = feature_table["adjustable"] & (
            feature_table["at_lower_bound"] | feature_table["at_upper_bound"]
        )
        return feature_table.loc[mask].copy()

    @staticmethod
    def _make_fixed_out_of_bounds(feature_table):
        mask = (~feature_table["adjustable"]) & feature_table["initial_out_of_bounds"]
        return feature_table.loc[mask].copy()

    @staticmethod
    def save_outputs(result, output_dir=Path("inverse_prediction/outputs")):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result.feature_table.to_csv(output_dir / "inverse_optimization_features.csv", index=False)
        result.history.to_csv(output_dir / "inverse_optimization_history.csv", index=False)

        summary_row = {}
        for idx, name in enumerate(TARGET_NAMES):
            summary_row[f"initial_{name}"] = result.initial_pred_raw[idx]
            summary_row[f"target_{name}"] = result.target_raw[idx]
            summary_row[f"final_{name}"] = result.final_pred_raw[idx]
            summary_row[f"{name}_error_to_target"] = (
                np.nan
                if np.isnan(result.target_raw[idx])
                else result.final_pred_raw[idx] - result.target_raw[idx]
            )
            summary_row[f"{name}_delta_from_initial"] = (
                result.final_pred_raw[idx] - result.initial_pred_raw[idx]
            )
        summary = pd.DataFrame([summary_row])
        summary.to_csv(output_dir / "inverse_optimization_summary.csv", index=False)

        fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(5 * len(TARGET_NAMES), 4))
        axes = np.atleast_1d(axes)
        for idx, name in enumerate(TARGET_NAMES):
            ax = axes[idx]
            ax.plot(result.history["step"], result.history[f"{name}_pred"], label=f"{name}_pred")
            if not np.isnan(result.target_raw[idx]):
                ax.axhline(
                    result.target_raw[idx],
                    color="r",
                    linestyle="--",
                    label=f"target_{name}",
                )
            ax.set_xlabel("Step")
            ax.set_ylabel(f"{name}_pred")
            ax.legend()
            ax.set_title(f"{name} convergence")

        fig.tight_layout()
        fig.savefig(output_dir / "inverse_opt_convergence.png", dpi=150)
        plt.close(fig)

        return {
            "feature_table": output_dir / "inverse_optimization_features.csv",
            "history": output_dir / "inverse_optimization_history.csv",
            "summary": output_dir / "inverse_optimization_summary.csv",
            "plot": output_dir / "inverse_opt_convergence.png",
        }
