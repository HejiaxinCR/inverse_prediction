from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .model import BASE_FEATURE_NAMES, FEATURE_NAMES, REFERENCE_FEATURE_NAMES, TARGET_NAMES
except ImportError:
    from model import BASE_FEATURE_NAMES, FEATURE_NAMES, REFERENCE_FEATURE_NAMES, TARGET_NAMES


SEED = 42
F = 9.64853321233100184 * 1e4  # C/mol
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TRAINING_FEATURE_BOUNDS = {
    "min": {
        "Tano_in": 48.442270771254144,
        "RHano_in": 0.50015479902274229,
        "Pano_in": 1.4422662282564611,
        "STC_ano": 1.5011135623466538,
        "XH2_dry": 0.8500180394346013,
        "Tcat_in": 45.958766933665999,
        "RHcat_in": 0.14023335493313549,
        "Pcat_in": 1.244253140981862,
        "STC_cat": 1.6010429682377392,
        "TCL_in": 50.020574883526699,
        "TCL_out": 50.452187279321038,
        "I": 30.15526482450781,
        "RH_ano_out_maxlim": 0.35947634371448461,
        "RH_cat_out_maxlim": 0.53507449800982676,
    },
    "max": {
        "Tano_in": 61.932676459753623,
        "RHano_in": 0.99973997905782364,
        "Pano_in": 2.3697264854477131,
        "STC_ano": 3.3987459173777284,
        "XH2_dry": 0.99993739959397399,
        "Tcat_in": 65.6397220574421,
        "RHcat_in": 0.17999530451455989,
        "Pcat_in": 2.1304806676940329,
        "STC_cat": 4.197819934538062,
        "TCL_in": 67.162501263812572,
        "TCL_out": 91.226090036546438,
        "I": 499.88146398417359,
        "RH_ano_out_maxlim": 1.0,
        "RH_cat_out_maxlim": 1.0,
    },
}


def resolve_existing_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        SCRIPT_DIR / path,
        SCRIPT_DIR.parent / path,
    ]
    if path.parts and path.parts[0] == SCRIPT_DIR.name:
        candidates.append(SCRIPT_DIR.joinpath(*path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def water_saturation_pressure(temp_c):
    """Approximate saturation pressure of water in Pa from temperature in Celsius."""
    a = 8.07131
    b = 1730.63
    c = 233.426
    pressure_mmhg = 10 ** (a - b / (c + temp_c))
    return pressure_mmhg * 133.322


def calculate_rh_outlet_reference(row):
    t_ano_in = row["Tano_in"]
    rh_ano_in = row["RHano_in"]
    p_ano = row["Pano_in"]
    p_cat = row["Pcat_in"]
    xh2_dry = row["XH2_dry"]
    current = row["I"]

    t_ano_out = row["TCL_out"]
    t_cat_in = row["Tcat_in"]
    rh_cat_in = row["RHcat_in"]
    t_cat_out = row["TCL_out"]
    stc_ano = row["STC_ano"]
    stc_cat = row["STC_cat"]

    n_h2_consumed = current / (2 * F)
    water_production = n_h2_consumed
    n_h2_in = n_h2_consumed * stc_ano
    n_h2_out = n_h2_in - n_h2_consumed

    p_total_ano = p_ano * 1e5
    p_sat_ano_in = water_saturation_pressure(t_ano_in)
    p_v_ano_in = rh_ano_in * p_sat_ano_in
    y_water_ano_in = p_v_ano_in / p_total_ano

    n_n2_ano_in = n_h2_in * (1.0 / xh2_dry - 1.0)
    n_water_ano_in = (n_h2_in + n_n2_ano_in) * y_water_ano_in / (1.0 - y_water_ano_in)
    n_water_ano_out = n_water_ano_in

    y_water_ano_out = n_water_ano_out / (n_water_ano_out + n_h2_out + n_n2_ano_in)
    p_v_ano_out = y_water_ano_out * p_total_ano
    p_sat_ano_out = water_saturation_pressure(t_ano_out)
    rh_ano_out = np.clip(p_v_ano_out / p_sat_ano_out, 0.0, 1.0)

    n_o2_consumed = current / (4 * F)
    n_o2_in = n_o2_consumed * stc_cat
    n_n2_cat_in = n_o2_in * 79 / 21
    n_o2_out = n_o2_in - n_o2_consumed
    n_dry_cat_out = n_o2_out + n_n2_cat_in

    p_total_cat = p_cat * 1e5
    p_sat_cat_in = water_saturation_pressure(t_cat_in)
    p_v_cat_in = rh_cat_in * p_sat_cat_in
    y_water_cat_in = p_v_cat_in / p_total_cat
    n_dry_cat_in = n_o2_in + n_n2_cat_in
    n_water_cat_in = n_dry_cat_in * y_water_cat_in / (1.0 - y_water_cat_in)
    n_water_cat_out = n_water_cat_in + water_production

    total_cat_out = n_water_cat_out + n_dry_cat_out
    y_water_cat_out = n_water_cat_out / total_cat_out if total_cat_out > 0 else 0.0
    p_v_cat_out = y_water_cat_out * p_total_cat
    p_sat_cat_out = water_saturation_pressure(t_cat_out)
    rh_cat_out = np.clip(p_v_cat_out / p_sat_cat_out, 0.0, 1.0)

    return pd.Series(
        {
            "RH_ano_out_maxlim": float(rh_ano_out),
            "RH_cat_out_maxlim": float(rh_cat_out),
        }
    )


def append_reference_features(frame):
    missing_reference = [name for name in REFERENCE_FEATURE_NAMES if name not in frame.columns]
    if not missing_reference:
        return frame.copy()

    missing_base = [name for name in BASE_FEATURE_NAMES if name not in frame.columns]
    if missing_base:
        raise ValueError(
            "Cannot derive reference feature columns because base columns are missing: "
            f"{missing_base}"
        )

    numeric_frame = frame.copy()
    for name in BASE_FEATURE_NAMES:
        numeric_frame[name] = pd.to_numeric(numeric_frame[name], errors="coerce")
    reference_df = numeric_frame.apply(calculate_rh_outlet_reference, axis=1)
    return pd.concat(
        [numeric_frame.reset_index(drop=True), reference_df.reset_index(drop=True)],
        axis=1,
    )


def load_model_data(path, source_name):
    path = resolve_existing_path(path)
    df = pd.read_csv(path)
    df = append_reference_features(df)
    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        raise ValueError(f"{path} is missing required feature columns: {missing_features}")

    available_target_cols = [col for col in TARGET_NAMES if col in df.columns]
    required_cols = FEATURE_NAMES + available_target_cols

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    df["source"] = source_name
    return df


def build_default_training_frame():
    return pd.DataFrame(
        [
            DEFAULT_TRAINING_FEATURE_BOUNDS["min"],
            DEFAULT_TRAINING_FEATURE_BOUNDS["max"],
        ],
        columns=FEATURE_NAMES,
    )


def rebuild_training_frame(train_path=None):
    if train_path is None:
        return build_default_training_frame()

    train_df = load_model_data(train_path, "train")
    train_df["flag_zero_output_with_positive_current"] = (
        train_df["I"].gt(0) & train_df["sa"].abs().le(1e-12) & train_df["ua"].abs().le(1e-12)
    )
    return (
        train_df.loc[~train_df["flag_zero_output_with_positive_current"]]
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
        .copy()
    )


def compute_feature_bounds(train_df):
    return train_df[FEATURE_NAMES].min(), train_df[FEATURE_NAMES].max()
