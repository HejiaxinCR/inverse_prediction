from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    from .inverse_optimizer import InverseOptimizer
    from .data_bounds import append_reference_features
    from .model import BASE_FEATURE_NAMES, FEATURE_NAMES
except ImportError:
    from inverse_optimizer import InverseOptimizer
    from data_bounds import append_reference_features
    from model import BASE_FEATURE_NAMES, FEATURE_NAMES


TARGET_NAMES = ["sa", "ua", "lambda_airin"]
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_cli_path(path, default_relative=None):
    if path is None:
        path = default_relative
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    if path.parts and path.parts[0] == SCRIPT_DIR.name:
        return SCRIPT_DIR.joinpath(*path.parts[1:])
    return SCRIPT_DIR / path


def parse_x0(raw):
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) not in {len(BASE_FEATURE_NAMES), len(FEATURE_NAMES)}:
        raise argparse.ArgumentTypeError(
            "--x0 must contain either "
            f"{len(BASE_FEATURE_NAMES)} base feature values or "
            f"{len(FEATURE_NAMES)} full feature values."
        )
    if len(values) == len(BASE_FEATURE_NAMES):
        frame = append_reference_features(pd.DataFrame([values], columns=BASE_FEATURE_NAMES))
        values = frame.loc[0, FEATURE_NAMES].to_list()
    return np.array(values, dtype=np.float32)


def load_x0_from_csv(path, row_index):
    frame = pd.read_csv(path)
    frame = coerce_feature_frame(frame)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"row_index must be in [0, {len(frame) - 1}], got {row_index}")
    return frame.loc[row_index, FEATURE_NAMES].to_numpy(dtype=np.float32)


def coerce_feature_frame(frame):
    frame = append_reference_features(frame)
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")
    for name in FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna(subset=FEATURE_NAMES).reset_index(drop=True)


def run_single_feature_forward_contribution(
    optimizer,
    x0_raw,
    x_result_raw,
    feature_names,
    sample_index=None,
):
    baseline_pred_raw = optimizer.predict_raw(x0_raw)
    final_pred_raw = optimizer.predict_raw(x_result_raw)
    initial_base_raw = np.asarray(x0_raw, dtype=np.float32)[optimizer.base_idx].copy()
    result_base_raw = np.asarray(x_result_raw, dtype=np.float32)[optimizer.base_idx].copy()
    rows = []

    for feature_name in feature_names:
        feature_idx = FEATURE_NAMES.index(feature_name)
        candidate_base_raw = initial_base_raw.copy()
        if feature_name in BASE_FEATURE_NAMES:
            base_idx = optimizer.base_feature_names.index(feature_name)
            candidate_base_raw[base_idx] = result_base_raw[base_idx]
        candidate_full_raw = optimizer._ensure_full_feature_array(candidate_base_raw)
        single_pred_raw = optimizer.predict_raw(candidate_full_raw)
        raw_min = float(optimizer.x_min_raw[feature_name])
        raw_max = float(optimizer.x_max_raw[feature_name])
        optimized_value = float(candidate_full_raw[feature_idx])
        row = {
            "controlled_feature": feature_name,
            "initial_feature_value": float(x0_raw[feature_idx]),
            "optimized_feature_value": optimized_value,
            "feature_delta": float(optimized_value - x0_raw[feature_idx]),
            "baseline_sa": float(baseline_pred_raw[0]),
            "single_feature_sa": float(single_pred_raw[0]),
            "sa_delta": float(single_pred_raw[0] - baseline_pred_raw[0]),
            "baseline_ua": float(baseline_pred_raw[1]),
            "single_feature_ua": float(single_pred_raw[1]),
            "ua_delta": float(single_pred_raw[1] - baseline_pred_raw[1]),
            "baseline_lambda_airin": float(baseline_pred_raw[2]),
            "single_feature_lambda_airin": float(single_pred_raw[2]),
            "lambda_airin_delta": float(single_pred_raw[2] - baseline_pred_raw[2]),
            "final_all_features_sa": float(final_pred_raw[0]),
            "final_all_features_ua": float(final_pred_raw[1]),
            "final_all_features_lambda_airin": float(final_pred_raw[2]),
            "at_lower_bound": bool(np.isclose(optimized_value, raw_min, rtol=0, atol=1e-6)),
            "at_upper_bound": bool(np.isclose(optimized_value, raw_max, rtol=0, atol=1e-6)),
            "hit_bound": bool(
                np.isclose(optimized_value, raw_min, rtol=0, atol=1e-6)
                or np.isclose(optimized_value, raw_max, rtol=0, atol=1e-6)
            ),
        }
        if sample_index is not None:
            row["sample_index"] = sample_index

        rows.append(row)

    column_order = [
        "sample_index",
        "controlled_feature",
        "initial_feature_value",
        "optimized_feature_value",
        "feature_delta",
        "baseline_sa",
        "single_feature_sa",
        "sa_delta",
        "baseline_ua",
        "single_feature_ua",
        "ua_delta",
        "baseline_lambda_airin",
        "single_feature_lambda_airin",
        "lambda_airin_delta",
        "final_all_features_sa",
        "final_all_features_ua",
        "final_all_features_lambda_airin",
        "at_lower_bound",
        "at_upper_bound",
        "hit_bound",
    ]
    analysis = pd.DataFrame(rows)
    existing_columns = [name for name in column_order if name in analysis.columns]
    return analysis.loc[:, existing_columns]


def run_batch(args, adjustable_names):
    batch_df = coerce_feature_frame(pd.read_csv(args.batch_csv))
    if args.batch_limit is not None:
        batch_df = batch_df.head(args.batch_limit).copy()
    if batch_df.empty:
        raise SystemExit("Batch CSV has no valid feature rows.")

    if "I" in adjustable_names:
        adjustable_names = [name for name in adjustable_names if name != "I"]
        print("Batch mode: removed I from adjustable variables, so current is fixed.")

    optimizer = InverseOptimizer(artifact_dir=args.artifact_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    optimized_rows = []
    feature_tables = []
    histories = []
    single_feature_tables = []
    batch_print_every = None if args.batch_print_every <= 0 else args.batch_print_every

    for row_index, row in batch_df.iterrows():
        x0_raw = row[FEATURE_NAMES].to_numpy(dtype=np.float32)
        result = optimizer.optimize(
            x0_raw=x0_raw,
            target_sa=args.target_sa,
            target_ua=args.target_ua,
            target_lambda_airin=args.target_lambda_airin,
            adjustable_names=adjustable_names,
            lr=args.lr,
            n_steps=args.steps,
            reg_weight=args.reg_weight,
            w_sa=args.w_sa,
            w_ua=args.w_ua,
            w_lambda_airin=args.w_lambda_airin,
            keep_sa_weight=args.keep_sa_weight,
            keep_ua_weight=args.keep_ua_weight,
            keep_lambda_airin_weight=args.keep_lambda_airin_weight,
            print_every=batch_print_every,
        )

        target_sa = result.target_raw[0]
        target_ua = result.target_raw[1]
        target_lambda_airin = result.target_raw[2]
        summary_rows.append(
            {
                "sample_index": row_index,
                "initial_model_sa": result.initial_pred_raw[0],
                "initial_model_ua": result.initial_pred_raw[1],
                "initial_model_lambda_airin": result.initial_pred_raw[2],
                "target_sa": target_sa,
                "target_ua": target_ua,
                "target_lambda_airin": target_lambda_airin,
                "final_model_sa": result.final_pred_raw[0],
                "final_model_ua": result.final_pred_raw[1],
                "final_model_lambda_airin": result.final_pred_raw[2],
                "sa_error_to_target": (
                    np.nan if np.isnan(target_sa) else result.final_pred_raw[0] - target_sa
                ),
                "ua_error_to_target": (
                    np.nan if np.isnan(target_ua) else result.final_pred_raw[1] - target_ua
                ),
                "lambda_airin_error_to_target": (
                    np.nan
                    if np.isnan(target_lambda_airin)
                    else result.final_pred_raw[2] - target_lambda_airin
                ),
                "initial_csv_sa": row["sa"] if "sa" in batch_df.columns else np.nan,
                "initial_csv_ua": row["ua"] if "ua" in batch_df.columns else np.nan,
                "initial_csv_lambda_airin": (
                    row["lambda_airin"] if "lambda_airin" in batch_df.columns else np.nan
                ),
                "initial_I": result.x_initial_raw[FEATURE_NAMES.index("I")],
                "optimized_I": result.x_result_raw[FEATURE_NAMES.index("I")],
                "touched_bounds": ",".join(result.touched_bounds["feature"].astype(str)),
                "fixed_out_of_bounds": ",".join(result.fixed_out_of_bounds["feature"].astype(str)),
            }
        )

        optimized_row = {"sample_index": row_index}
        optimized_row.update(
            {name: result.x_result_raw[idx] for idx, name in enumerate(FEATURE_NAMES)}
        )
        optimized_row.update(
            {
                "final_model_sa": result.final_pred_raw[0],
                "final_model_ua": result.final_pred_raw[1],
                "final_model_lambda_airin": result.final_pred_raw[2],
            }
        )
        optimized_rows.append(optimized_row)

        feature_table = result.feature_table.copy()
        feature_table.insert(0, "sample_index", row_index)
        feature_tables.append(feature_table)

        history = result.history.copy()
        history.insert(0, "sample_index", row_index)
        histories.append(history)

        if args.single_feature_analysis:
            single_feature_tables.append(
                run_single_feature_forward_contribution(
                    optimizer=optimizer,
                    x0_raw=x0_raw,
                    x_result_raw=result.x_result_raw,
                    feature_names=adjustable_names,
                    sample_index=row_index,
                )
            )

        if (row_index + 1) % max(args.batch_status_every, 1) == 0 or row_index == len(batch_df) - 1:
            print(
                f"optimized {row_index + 1}/{len(batch_df)} rows; "
                f"final sa={result.final_pred_raw[0]:.6f}, "
                f"ua={result.final_pred_raw[1]:.6f}, "
                f"lambda_airin={result.final_pred_raw[2]:.6f}"
            )

    summary = pd.DataFrame(summary_rows)
    optimized_samples = pd.DataFrame(optimized_rows)
    feature_changes = pd.concat(feature_tables, ignore_index=True)
    history_all = pd.concat(histories, ignore_index=True)
    single_feature_analysis = (
        pd.concat(single_feature_tables, ignore_index=True)
        if single_feature_tables
        else pd.DataFrame()
    )

    summary_path = args.output_dir / "batch_inverse_optimization_summary.csv"
    optimized_path = args.output_dir / "batch_inverse_optimization_optimized_samples.csv"
    feature_path = args.output_dir / "batch_inverse_optimization_feature_changes.csv"
    history_path = args.output_dir / "batch_inverse_optimization_history.csv"
    single_feature_path = args.output_dir / "batch_single_feature_forward_contribution.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    optimized_samples.to_csv(optimized_path, index=False, encoding="utf-8-sig")
    feature_changes.to_csv(feature_path, index=False, encoding="utf-8-sig")
    history_all.to_csv(history_path, index=False, encoding="utf-8-sig")
    if not single_feature_analysis.empty:
        single_feature_analysis.to_csv(single_feature_path, index=False, encoding="utf-8-sig")

    print("\nSaved batch outputs:")
    print(f"  summary:           {summary_path}")
    print(f"  optimized samples: {optimized_path}")
    print(f"  feature changes:   {feature_path}")
    print(f"  history:           {history_path}")
    if not single_feature_analysis.empty:
        print(f"  single feature:    {single_feature_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Optimize PEMFC operating inputs against a frozen MLP surrogate."
    )
    parser.add_argument("--target-sa", type=float, default=None)
    parser.add_argument("--target-ua", type=float, default=None)
    parser.add_argument("--target-lambda-airin", type=float, default=None)
    parser.add_argument(
        "--x0",
        type=parse_x0,
        default=None,
        help=(
            "Either 12 base feature values or 14 full feature values in model input order. "
            "If 12 are provided, outlet RH max-limit features are derived automatically."
        ),
    )
    parser.add_argument("--x0-csv", type=Path, default=None)
    parser.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="CSV with feature columns; optimize every row in batch mode.",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--batch-limit", type=int, default=None)
    parser.add_argument("--batch-status-every", type=int, default=10)
    parser.add_argument(
        "--adjustable",
        type=str,
        default="Tano_in,RHano_in,Pano_in,STC_ano,XH2_dry,Tcat_in,RHcat_in,Pcat_in,STC_cat,TCL_in,FR_CL,I",
        help="Comma-separated feature names to optimize.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reg-weight", type=float, default=0.01)
    parser.add_argument("--w-sa", type=float, default=1.0)
    parser.add_argument("--w-ua", type=float, default=1.0)
    parser.add_argument("--w-lambda-airin", type=float, default=1.0)
    parser.add_argument(
        "--keep-sa-weight",
        type=float,
        default=0.0,
        help="Penalty for keeping sa near its original prediction when --target-sa is omitted.",
    )
    parser.add_argument(
        "--keep-ua-weight",
        type=float,
        default=0.0,
        help="Penalty for keeping ua near its original prediction when --target-ua is omitted.",
    )
    parser.add_argument(
        "--keep-lambda-airin-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty for keeping lambda_airin near its original prediction when "
            "--target-lambda-airin is omitted."
        ),
    )
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument(
        "--batch-print-every",
        type=int,
        default=0,
        help="Per-row optimizer progress interval in batch mode; 0 disables per-step logs.",
    )
    parser.add_argument(
        "--single-feature-analysis",
        action="store_true",
        help=(
            "Run forward contribution analysis once per adjustable feature by replaying that "
            "feature's initial-to-optimized change while keeping all other features at initial values."
        ),
    )
    args = parser.parse_args()
    args.artifact_dir = resolve_cli_path(args.artifact_dir, "module_value")
    args.output_dir = resolve_cli_path(args.output_dir, "outputs")

    input_modes = [args.x0 is not None, args.x0_csv is not None, args.batch_csv is not None]
    if sum(input_modes) != 1:
        raise SystemExit("Provide exactly one of --x0, --x0-csv, or --batch-csv.")
    if (
        args.target_sa is None
        and args.target_ua is None
        and args.target_lambda_airin is None
    ):
        raise SystemExit(
            "Provide at least one of --target-sa, --target-ua, or --target-lambda-airin."
        )

    adjustable_names = [part.strip() for part in args.adjustable.split(",") if part.strip()]

    if args.batch_csv is not None:
        run_batch(args, adjustable_names)
        return

    x0_raw = args.x0 if args.x0 is not None else load_x0_from_csv(args.x0_csv, args.row_index)

    optimizer = InverseOptimizer(artifact_dir=args.artifact_dir)
    result = optimizer.optimize(
        x0_raw=x0_raw,
        target_sa=args.target_sa,
        target_ua=args.target_ua,
        target_lambda_airin=args.target_lambda_airin,
        adjustable_names=adjustable_names,
        lr=args.lr,
        n_steps=args.steps,
        reg_weight=args.reg_weight,
        w_sa=args.w_sa,
        w_ua=args.w_ua,
        w_lambda_airin=args.w_lambda_airin,
        keep_sa_weight=args.keep_sa_weight,
        keep_ua_weight=args.keep_ua_weight,
        keep_lambda_airin_weight=args.keep_lambda_airin_weight,
        print_every=args.print_every,
    )
    outputs = optimizer.save_outputs(result, output_dir=args.output_dir)
    single_feature_path = None
    single_feature_analysis = None
    if args.single_feature_analysis:
        single_feature_analysis = run_single_feature_forward_contribution(
            optimizer=optimizer,
            x0_raw=x0_raw,
            x_result_raw=result.x_result_raw,
            feature_names=adjustable_names,
        )
        single_feature_path = args.output_dir / "single_feature_forward_contribution.csv"
        single_feature_analysis.to_csv(single_feature_path, index=False, encoding="utf-8-sig")

    print("\nInitial prediction:")
    print(
        "  "
        f"sa={result.initial_pred_raw[0]:.6f}, "
        f"ua={result.initial_pred_raw[1]:.6f}, "
        f"lambda_airin={result.initial_pred_raw[2]:.6f}"
    )
    print("Final prediction:")
    print(
        "  "
        f"sa={result.final_pred_raw[0]:.6f}, "
        f"ua={result.final_pred_raw[1]:.6f}, "
        f"lambda_airin={result.final_pred_raw[2]:.6f}"
    )
    print("Target:")
    target_sa_text = "None" if np.isnan(result.target_raw[0]) else f"{result.target_raw[0]:.6f}"
    target_ua_text = "None" if np.isnan(result.target_raw[1]) else f"{result.target_raw[1]:.6f}"
    target_lambda_airin_text = (
        "None" if np.isnan(result.target_raw[2]) else f"{result.target_raw[2]:.6f}"
    )
    print(
        "  "
        f"sa={target_sa_text}, "
        f"ua={target_ua_text}, "
        f"lambda_airin={target_lambda_airin_text}"
    )
    print("\nFeature changes:")
    print(result.feature_table.to_string(index=False))

    if single_feature_analysis is not None and not single_feature_analysis.empty:
        print("\nSingle-feature forward contribution:")
        summary_cols = [
            "controlled_feature",
            "initial_feature_value",
            "optimized_feature_value",
            "feature_delta",
            "baseline_sa",
            "single_feature_sa",
            "sa_delta",
            "baseline_ua",
            "single_feature_ua",
            "ua_delta",
            "baseline_lambda_airin",
            "single_feature_lambda_airin",
            "lambda_airin_delta",
            "hit_bound",
        ]
        existing_summary_cols = [name for name in summary_cols if name in single_feature_analysis.columns]
        print(
            single_feature_analysis.loc[:, existing_summary_cols]
            .sort_values(by="sa_delta", ascending=True, na_position="last")
            .to_string(index=False)
        )

    if len(result.touched_bounds) > 0:
        print("\nAdjustable variables touching training bounds:")
        print(result.touched_bounds[["feature", "optimized", "train_min", "train_max"]].to_string(index=False))
    if len(result.fixed_out_of_bounds) > 0:
        print("\nFixed variables whose initial values are outside training bounds:")
        print(
            result.fixed_out_of_bounds[
                ["feature", "initial", "train_min", "train_max"]
            ].to_string(index=False)
        )

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    if single_feature_path is not None:
        print(f"  single_feature_forward_contribution: {single_feature_path}")


if __name__ == "__main__":
    main()
