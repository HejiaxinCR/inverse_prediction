# Inverse Prediction 使用说明

本目录包含两个主要入口：

- `run_inverse_optimization.py`：逆向优化工况。给定目标 `sa`、`ua`、`lambda_airin`，反推可调工况参数。
- `batch_predict.py`：批量正向预测。读取一个或多个 `.csv` 数据集，输出 `pred_sa`、`pred_ua`、`pred_lambda_airin`。

## Python 环境配置

推荐使用独立的 Python 环境运行，避免和系统 Python 或其他项目的依赖冲突。建议 Python 版本使用 `3.10` 或 `3.11`。

下面两种方式任选一种即可：

- 方案 A：使用 conda 管理环境。
- 方案 B：使用 Python 自带的 `venv` 管理环境。

### 1A. 使用 conda 创建并激活环境

在 PowerShell 或 Anaconda Prompt 中运行：

```powershell
conda create -n pemfc-inverse python=3.10 -y
conda activate pemfc-inverse
```

如果已经有自己的环境，也可以直接激活已有环境：

```powershell
conda activate <你的环境名>
```

### 1B. 使用 venv 创建并激活环境

如果不使用 conda，也可以用 Python 自带的 `venv`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止执行脚本，可以在当前终端临时放开执行策略后再激活：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

以后每次重新打开终端，只需要在本目录重新激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 Python 包

本目录脚本依赖以下 Python 包：

```powershell
python -m pip install numpy pandas matplotlib scikit-learn torch
```

如果需要使用 GPU，请根据本机 CUDA 版本到 PyTorch 官网选择对应安装命令；如果不确定，先使用上面的 CPU 版本即可。

### 3. 验证环境

安装完成后，可以运行下面的命令检查依赖是否可正常导入：

```powershell
python -c "import numpy, pandas, matplotlib, sklearn, torch; print('Python env OK'); print('torch cuda:', torch.cuda.is_available())"
```

如果使用 conda，以后每次重新打开终端运行脚本前，先进入项目目录并激活环境：

```powershell
conda activate pemfc-inverse
```

## 路径约定

下面的示例命令都假设先进入当前脚本目录 `inverse_prediction/`：

```powershell
cd '<你的项目根目录>\inverse_prediction'

python '.\batch_predict.py' `
  --input '.\data\opcon_V1_83p1_300p8_480p8_converted.csv' `
  --output '.\outputs\batch_predictions\opcon_predictions.csv' `
  --device cpu
```

逆向优化脚本示例：

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.006 `
  --x0 '66.9,0.5335,2.362,1.689,0.9403,64.5,0.147,1.792,1.687,67.1,0.03,480.8' `
  --adjustable Tano_in,RHano_in,Pano_in,STC_ano,XH2_dry,Tcat_in,RHcat_in,Pcat_in,STC_cat,TCL_in,FR_CL `
  --output-dir '.\outputs\inverse_from_x0'
```

## 模型文件

脚本默认会自动查找 `module_value/` 或 `inverse_prediction/module_value/`，并加载以下模型产物：

- `mlp_surrogate_state_dict.pt`
- `normalizer_x.pkl`
- `normalizer_y.pkl`

如果模型文件放在其他位置，可以显式指定：

```powershell
--artifact-dir '你的模型目录'
```

冻结的 MLP surrogate 预测 3 个输出：

```text
sa, ua, lambda_airin
```

模型输入共 14 个特征：

```text
Tano_in,RHano_in,Pano_in,STC_ano,XH2_dry,Tcat_in,RHcat_in,Pcat_in,STC_cat,TCL_in,FR_CL,I,RH_ano_out_maxlim,RH_cat_out_maxlim
```

如果 CSV 或 `--x0` 只提供前 12 个基础工况参数，代码会自动计算：

```text
RH_ano_out_maxlim,RH_cat_out_maxlim
```

这两个参考特征只用于模型输入，`batch_predict.py` 的输出文件不会保留它们。

## 批量预测 batch_predict.py

用途：读取 CSV 工况数据，批量预测 `sa`、`ua`、`lambda_airin`。

### 预测单个 CSV

```powershell
python '.\batch_predict.py' `
  --input 'data\opcon_V1_83p1_300p8_480p8_converted.csv' `
  --output 'outputs\batch_predictions\opcon_V1_83p1_300p8_480p8_converted_predictions.csv' `
  --summary 'outputs\batch_predictions\opcon_V1_83p1_300p8_480p8_converted_summary.csv' `
  --device cpu
```

### 预测整个文件夹

默认会递归读取文件夹下所有 `.csv`：

```powershell
python '.\batch_predict.py' `
  --input 'data' `
  --output 'outputs\batch_predictions' `
  --device cpu
```

如果只想读取当前文件夹，不递归子文件夹：

```powershell
python '.\batch_predict.py' `
  --input 'data' `
  --output 'outputs\batch_predictions' `
  --no-recursive `
  --device cpu
```

### 输出文件说明

输出 CSV 会保留原始工况列，并新增：

```text
pred_sa,pred_ua,pred_lambda_airin
```

如果原始 CSV 里有真实数值形式的 `sa`、`ua` 或 `lambda_airin`，会额外输出对应误差列：

```text
error_sa,error_ua,error_lambda_airin
```

如果原始 CSV 里没有真实目标值，或者目标列只是空值/单位行，则输出文件不会保留这些空目标列，也不会生成 `error_*` 字段。

`pred_sa` 在预测后会做非负截断：

```text
pred_sa = max(pred_sa, 0)
```

### FR_CL 缺失时的处理

推荐先直接把 `FR_CL` 填入原始 CSV，再运行预测脚本。例如当前 opcon 数据中：

```text
83.1 A   -> FR_CL = 0.015
300.8 A  -> FR_CL = 0.02
480.8 A  -> FR_CL = 0.03
```

如果 CSV 里 `FR_CL` 仍然缺失，对应行会被跳过，预测值输出为 NaN。

脚本也保留了一个可选命令行参数，用于临时按电流档位填充缺失的 `FR_CL`：

```powershell
python '.\batch_predict.py' `
  --input 'data\opcon_V1_83p1_300p8_480p8_converted.csv' `
  --output 'outputs\batch_predictions\opcon_predictions.csv' `
  --fr-cl-by-current '83:0.015,300:0.02,480:0.03' `
  --device cpu
```

## 逆向优化 run_inverse_optimization.py

用途：给定目标输出，优化可调输入工况，使模型预测尽量接近目标。

至少需要指定一个目标：

```text
--target-sa
--target-ua
--target-lambda-airin
```

可以通过 `--adjustable` 指定哪些工况参数允许被优化。多个参数用英文逗号分隔：

```text
--adjustable I,STC_ano,RHano_in
```

### 从 CSV 选择一行作为初始工况

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.006 `
  --x0-csv 'data\database_LHS_demo_test_0709.csv' `
  --row-index 0 `
  --adjustable I,STC_ano,RHano_in `
  --output-dir 'outputs\inverse_single_sa'
```

### 只优化 ua，同时约束 sa 不要偏离初始预测太多

```powershell
python '.\run_inverse_optimization.py' `
  --target-ua 2.0 `
  --keep-sa-weight 1.0 `
  --x0-csv 'data\database_LHS_demo_test_0709.csv' `
  --row-index 0 `
  --adjustable I,STC_ano,RHano_in `
  --output-dir 'outputs\inverse_single_ua'
```

### 同时优化 sa、ua、lambda_airin

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.006 `
  --target-ua 2.0 `
  --target-lambda-airin 2.3 `
  --x0-csv 'data\database_LHS_demo_test_0709.csv' `
  --row-index 0 `
  --adjustable I,STC_ano,RHano_in `
  --output-dir 'outputs\inverse_single_all'
```

### 直接输入 12 个基础工况参数

顺序必须是：

```text
Tano_in,RHano_in,Pano_in,STC_ano,XH2_dry,Tcat_in,RHcat_in,Pcat_in,STC_cat,TCL_in,FR_CL,I
```

示例：

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.028 `
  --x0 '66.9,0.5335,2.362,1.689,0.9403,64.5,0.147,1.792,1.687,67.1,0.03,480.8' `
  --adjustable Tano_in,RHano_in,Pano_in,STC_ano,XH2_dry,Tcat_in,RHcat_in,Pcat_in,STC_cat,TCL_in,FR_CL `
  --output-dir 'outputs\inverse_from_x0'
```

### 批量逆向优化

对 CSV 中每一行都做逆向优化：

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.006 `
  --batch-csv 'data\database_LHS_demo_test_0709.csv' `
  --adjustable I,STC_ano,RHano_in `
  --batch-print-every 0 `
  --output-dir 'outputs\inverse_batch_sa'
```

只跑前 10 行：

```powershell
python '.\run_inverse_optimization.py' `
  --target-sa 0.006 `
  --batch-csv 'data\database_LHS_demo_test_0709.csv' `
  --batch-limit 10 `
  --adjustable I,STC_ano,RHano_in `
  --batch-print-every 0 `
  --output-dir 'outputs\inverse_batch_sa_10rows'
```

### 常用优化参数

- `--lr`：优化学习率，默认值由脚本内部设置。
- `--steps`：优化步数。
- `--reg-weight`：输入变化的正则惩罚，越大越不愿意偏离初始工况。
- `--w-sa`、`--w-ua`、`--w-lambda-airin`：目标误差权重。
- `--keep-sa-weight`、`--keep-ua-weight`、`--keep-lambda-airin-weight`：未指定目标时，让该输出尽量保持初始预测值的权重。
- `--single-feature-analysis`：输出单变量贡献分析。

### 逆向优化输出

单行优化时，输出目录会包含：

- `inverse_optimization_summary.csv`
- `inverse_optimization_features.csv`
- `inverse_optimization_history.csv`
- `inverse_opt_convergence.png`
- `single_feature_forward_contribution.csv`，仅在使用 `--single-feature-analysis` 时生成

批量优化时，输出目录会包含：

- `batch_inverse_optimization_summary.csv`
- `batch_inverse_optimization_optimized_samples.csv`
- `batch_inverse_optimization_history.csv`
- `batch_inverse_optimization_feature_changes.csv`

## 快速检查命令

查看 `batch_predict.py` 参数：

```powershell
python '.\batch_predict.py' --help
```

查看 `run_inverse_optimization.py` 参数：

```powershell
python '.\run_inverse_optimization.py' --help
```

