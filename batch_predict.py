from pathlib import Path
import argparse
import pickle

import numpy as np
import pandas as pd
import torch

try:
    from .data_bounds import append_reference_features
    from .model import FEATURE_NAMES, REFERENCE_FEATURE_NAMES, TARGET_NAMES, MLPSurrogate
except ImportError:
    from data_bounds import append_reference_features
    from model import FEATURE_NAMES, REFERENCE_FEATURE_NAMES, TARGET_NAMES, MLPSurrogate


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "module_value"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "batch_predictions"
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_ARTIFACT_FILES = (
    "normalizer_x.pkl",
    "normalizer_y.pkl",
    "mlp_surrogate_state_dict.pt",
)
CSV_UNITS = {
    "Tano_in": "℃",
    "RHano_in": "-",
    "Pano_in": "bar",
    "STC_ano": "-",
    "XH2_dry": "-",
    "Tcat_in": "℃",
    "RHcat_in": "-",
    "Pcat_in": "bar",
    "STC_cat": "-",
    "TCL_in": "℃",
    "TCL_out": "℃",
    "FR_CL": "g/(s cm^2)",
    "I": "A",
    "sa": "-",
    "ua": "m/s",
    "lambda_airin": "-",
    "lambda_airout": "-",
}


def unit_for_column(column):
    if column.startswith("pred_"):
        return CSV_UNITS.get(column.removeprefix("pred_"), "")
    if column.startswith("error_"):
        return CSV_UNITS.get(column.removeprefix("error_"), "")
    return CSV_UNITS.get(column, "")


def normalize_unit(value):
    return str(value).strip().replace("°C", "℃").replace("cm²", "cm^2")


def is_unit_row(row):
    comparable_columns = [col for col in row.index if col in CSV_UNITS]
    if not comparable_columns:
        return False
    return all(normalize_unit(row[col]) == CSV_UNITS[col] for col in comparable_columns)


def drop_unit_row(frame):
    if frame.empty:
        return frame
    if is_unit_row(frame.iloc[0]):
        return frame.iloc[1:].reset_index(drop=True).copy()
    return frame


def to_csv_with_unit_row(frame, path):
    unit_row = pd.DataFrame(
        [[unit_for_column(column) for column in frame.columns]],
        columns=frame.columns,
    )
    pd.concat([unit_row, frame], ignore_index=True).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def candidate_paths(path):
    path = Path(path)
    if path.is_absolute():
        return [path]

    candidates = [Path.cwd() / path]
    if path.parts and path.parts[0] == SCRIPT_DIR.name:
        candidates.append(SCRIPT_DIR.joinpath(*path.parts[1:]))
    candidates.extend([SCRIPT_DIR / path, SCRIPT_DIR.parent / path])

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
    return unique_candidates


def resolve_existing_path(path, label):
    candidates = candidate_paths(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} does not exist. Checked:\n  {checked}")


def resolve_output_path(path, default_path=None):
    if path is None:
        return Path(default_path)
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    if path.parts and path.parts[0] == SCRIPT_DIR.name:
        return SCRIPT_DIR.joinpath(*path.parts[1:])
    return SCRIPT_DIR / path


def resolve_artifact_dir(artifact_dir):
    candidates = candidate_paths(artifact_dir)
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_ARTIFACT_FILES):
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    required = ", ".join(REQUIRED_ARTIFACT_FILES)
    raise FileNotFoundError(f"Could not find model artifacts ({required}). Checked:\n  {checked}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_surrogate(artifact_dir, device):
    artifact_dir = resolve_artifact_dir(artifact_dir)
    normalizer_x = load_pickle(artifact_dir / "normalizer_x.pkl")
    normalizer_y = load_pickle(artifact_dir / "normalizer_y.pkl")

    model = MLPSurrogate(input_dim=len(FEATURE_NAMES), output_dim=len(TARGET_NAMES)).to(device)
    state_dict = torch.load(
        artifact_dir / "mlp_surrogate_state_dict.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, normalizer_x, normalizer_y


def parse_fr_cl_by_current(raw):
    mapping = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        current, fr_cl = item.split(":", 1)
        mapping[float(current.strip())] = float(fr_cl.strip())
    if not mapping:
        raise argparse.ArgumentTypeError("--fr-cl-by-current must contain at least one I:FR_CL pair.")
    return mapping


def fill_fr_cl_by_current(frame, mapping):
    if not mapping:
        return frame

    frame = frame.copy()
    current_values = np.array(list(mapping.keys()), dtype=np.float32)
    fr_cl_values = np.array([mapping[current] for current in mapping.keys()], dtype=np.float32)
    missing_fr_cl = frame["FR_CL"].isna() & frame["I"].notna()
    if not missing_fr_cl.any():
        return frame

    currents = frame.loc[missing_fr_cl, "I"].to_numpy(dtype=np.float32)
    nearest_idx = np.abs(currents[:, None] - current_values[None, :]).argmin(axis=1)
    frame.loc[missing_fr_cl, "FR_CL"] = fr_cl_values[nearest_idx]
    return frame


def coerce_feature_frame(frame, fr_cl_by_current=None):
    frame = append_reference_features(frame)
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")

    for name in FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = fill_fr_cl_by_current(frame, fr_cl_by_current)
    return frame


def predict_frame(frame, model, normalizer_x, normalizer_y, device, batch_size, fr_cl_by_current=None):
    frame = coerce_feature_frame(frame, fr_cl_by_current=fr_cl_by_current)
    valid_mask = frame[FEATURE_NAMES].notna().all(axis=1)
    predictions = np.full((len(frame), len(TARGET_NAMES)), np.nan, dtype=np.float32)

    if valid_mask.any():
        x_raw = frame.loc[valid_mask, FEATURE_NAMES].to_numpy(dtype=np.float32)
        x_norm = normalizer_x.transform(x_raw).astype(np.float32)

        pred_batches = []
        with torch.no_grad():
            for start in range(0, len(x_norm), batch_size):
                batch = torch.tensor(
                    x_norm[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                pred_batches.append(model(batch).cpu().numpy())

        pred_norm = np.vstack(pred_batches)
        pred_raw = normalizer_y.inverse_transform(pred_norm).astype(np.float32)
        predictions[valid_mask.to_numpy()] = pred_raw

    output = frame.copy()
    for idx, name in enumerate(TARGET_NAMES):
        pred_col = f"pred_{name}"
        output[pred_col] = predictions[:, idx]
        if name == "sa":
            output[pred_col] = np.maximum(output[pred_col], 0.0)
        if name in output.columns:
            actual = pd.to_numeric(output[name], errors="coerce")
            if actual.notna().any():
                output[f"error_{name}"] = output[pred_col] - actual
            else:
                output = output.drop(columns=[name])
    output = output.drop(columns=list(REFERENCE_FEATURE_NAMES), errors="ignore")
    return output, int(valid_mask.sum()), int((~valid_mask).sum())


def iter_csv_files(input_path, pattern, recursive):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(path for path in globber(pattern) if path.is_file())


def output_path_for(input_file, input_root, output_arg):
    input_file = Path(input_file)
    if output_arg is not None:
        output_arg = Path(output_arg)
        if input_file.is_file() and output_arg.suffix.lower() == ".csv":
            return output_arg
        output_root = output_arg
    else:
        output_root = DEFAULT_OUTPUT_DIR

    if input_root.is_file():
        relative = Path(f"{input_file.stem}_predictions.csv")
    else:
        relative = input_file.relative_to(input_root)
        relative = relative.with_name(f"{relative.stem}_predictions.csv")
    return output_root / relative


def run(args):
    input_path = resolve_existing_path(args.input, "Input path")
    output_arg = resolve_output_path(args.output) if args.output else None
    summary_arg = resolve_output_path(args.summary) if args.summary else None

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, normalizer_x, normalizer_y = load_surrogate(args.artifact_dir, device)

    csv_files = iter_csv_files(input_path, args.pattern, args.recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_path}")

    summary_rows = []
    for csv_file in csv_files:
        frame = drop_unit_row(pd.read_csv(csv_file))
        predicted, valid_rows, invalid_rows = predict_frame(
            frame,
            model,
            normalizer_x,
            normalizer_y,
            device,
            args.batch_size,
            fr_cl_by_current=args.fr_cl_by_current,
        )
        out_path = output_path_for(csv_file, input_path, output_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_csv_with_unit_row(predicted, out_path)
        summary_rows.append(
            {
                "input_csv": str(csv_file),
                "output_csv": str(out_path),
                "rows": len(frame),
                "predicted_rows": valid_rows,
                "skipped_rows": invalid_rows,
            }
        )
        print(f"saved {out_path} ({valid_rows}/{len(frame)} rows predicted)")

    summary = pd.DataFrame(summary_rows)
    if summary_arg:
        summary_path = summary_arg
    elif output_arg and output_arg.suffix.lower() == ".csv":
        summary_path = output_arg.with_name("batch_prediction_summary.csv")
    else:
        summary_path = (output_arg if output_arg else DEFAULT_OUTPUT_DIR) / "batch_prediction_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"summary saved {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch predict sa, ua, and lambda_airin from CSV feature datasets."
    )
    parser.add_argument("--input", required=True, help="CSV file or folder containing CSV files.")
    parser.add_argument(
        "--output",
        help=(
            "Output CSV path for a single input file, or output folder for folder input. "
            "Default: inverse_prediction/outputs/batch_predictions."
        ),
    )
    parser.add_argument("--summary", help="Optional path for the batch summary CSV.")
    parser.add_argument("--pattern", default="*.csv", help="CSV filename pattern for folder input.")
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only read CSV files directly inside the input folder.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--fr-cl-by-current",
        type=parse_fr_cl_by_current,
        help="Fill missing FR_CL by nearest current, for example: 83:0.015,300:0.02,480:0.03",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Prediction device.")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.set_defaults(recursive=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
