import torch.nn as nn


BASE_FEATURE_NAMES = [
    "Tano_in",
    "RHano_in",
    "Pano_in",
    "STC_ano",
    "XH2_dry",
    "Tcat_in",
    "RHcat_in",
    "Pcat_in",
    "STC_cat",
    "TCL_in",
    # "TCL_out",
    "FR_CL",
    "I",
]
REFERENCE_FEATURE_NAMES = [
    "RH_ano_out_maxlim",
    "RH_cat_out_maxlim",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + REFERENCE_FEATURE_NAMES

TARGET_NAMES = ["sa", "ua", "lambda_airin"]


class MLPSurrogate(nn.Module):
    def __init__(self, input_dim=len(FEATURE_NAMES), output_dim=3, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)
