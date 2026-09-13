from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================
# 基本设置
# =========================
RANDOM_STATE = 82
LABEL_COLUMN = "fc/MPa"

# 是否保存评价结果
SAVE_RESULTS = True

seed_folder = Path("output") / f"seed_{RANDOM_STATE}"
data_folder = seed_folder / "data"
model_folder = seed_folder / "ag_model"
metrics_folder = seed_folder / "metrics"

test_data_path = data_folder / "test_data.csv"
x_test_path = data_folder / "X_test.csv"
y_test_path = data_folder / "y_test.csv"


# =========================
# 检查必要文件
# =========================
required_paths = [
    test_data_path,
    x_test_path,
    y_test_path,
    model_folder,
]

missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    missing_text = "\n".join(str(path) for path in missing_paths)
    raise FileNotFoundError(f"以下文件或文件夹不存在：\n{missing_text}")


# =========================
# 读取测试数据
# =========================
X_test = pd.read_csv(x_test_path, index_col=0)

y_test_df = pd.read_csv(y_test_path, index_col=0)

if y_test_df.empty:
    raise ValueError("y_test.csv 中没有数据。")

if LABEL_COLUMN in y_test_df.columns:
    y_test = pd.to_numeric(
        y_test_df[LABEL_COLUMN],
        errors="coerce",
    )
else:
    # 如果没有指定的标签列，就使用第一列
    y_test = pd.to_numeric(
        y_test_df.iloc[:, 0],
        errors="coerce",
    )
    print(
        f"警告：y_test.csv 中没有找到列 {LABEL_COLUMN!r}，"
        f"已使用第一列 {y_test_df.columns[0]!r}。"
    )


# =========================
# 对齐X_test与y_test
# =========================
common_index = X_test.index.intersection(y_test.index)

if len(common_index) == 0:
    raise ValueError("X_test 与 y_test 之间没有相同的样本索引。")

if len(common_index) != len(X_test) or len(common_index) != len(y_test):
    print(
        "警告：X_test 与 y_test 的索引没有完全一致，"
        "将只评价共同索引对应的样本。"
    )

X_test = X_test.loc[common_index].copy()
y_test = y_test.loc[common_index].copy()


# =========================
# 加载AutoGluon模型
# =========================
predictor = TabularPredictor.load(model_folder)

model_names = predictor.model_names()

if not model_names:
    raise ValueError("ag_model 中没有找到可用模型。")

print(f"\n找到 {len(model_names)} 个模型：")
for model_name in model_names:
    print(f"  - {model_name}")


# =========================
# 计算所有模型的指标
# =========================
results = []
prediction_results = pd.DataFrame(index=common_index)

for model_name in model_names:
    try:
        # 指定模型进行预测
        y_pred = predictor.predict(
            X_test,
            model=model_name,
        )

        # 确保预测结果与测试集索引一致
        y_pred = pd.Series(
            np.asarray(y_pred).reshape(-1),
            index=X_test.index,
            name=model_name,
        )

        y_pred = pd.to_numeric(
            y_pred,
            errors="coerce",
        )

        prediction_results[model_name] = y_pred

        # 去掉真实值或预测值为缺失值的样本
        valid_mask = y_test.notna() & y_pred.notna()

        y_true_valid = y_test.loc[valid_mask].to_numpy(dtype=float)
        y_pred_valid = y_pred.loc[valid_mask].to_numpy(dtype=float)

        sample_count = len(y_true_valid)

        if sample_count < 2:
            print(
                f"{model_name}：有效样本数量少于2，"
                "无法可靠计算R²，已跳过。"
            )
            continue

        # R²
        r2 = r2_score(
            y_true_valid,
            y_pred_valid,
        )

        # MSE与RMSE
        mse = mean_squared_error(
            y_true_valid,
            y_pred_valid,
        )
        rmse = np.sqrt(mse)

        # MAE
        mae = mean_absolute_error(
            y_true_valid,
            y_pred_valid,
        )

        # MAPE：排除真实值为0的样本
        non_zero_mask = y_true_valid != 0

        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (
                            y_true_valid[non_zero_mask]
                            - y_pred_valid[non_zero_mask]
                        )
                        / y_true_valid[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        # nRMSE：使用该模型有效观测值的极差归一化
        y_range = y_true_valid.max() - y_true_valid.min()

        if y_range > 0:
            nrmse = rmse / y_range * 100
        else:
            nrmse = np.nan

        results.append(
            {
                "Model": model_name,
                "Sample Count": sample_count,
                "R²": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE (%)": mape,
                "nRMSE (%)": nrmse,
            }
        )

        print(f"{model_name}：评价完成。")

    except Exception as error:
        print(f"{model_name}：评价失败，原因：{error}")


# =========================
# 整理并显示结果
# =========================
if not results:
    raise RuntimeError("没有任何模型成功完成评价。")

results_df = pd.DataFrame(results)

# R²越大越好
results_df = results_df.sort_values(
    by="R²",
    ascending=False,
).reset_index(drop=True)

display_results = results_df.copy()

numeric_columns = [
    "R²",
    "RMSE",
    "MAE",
    "MAPE (%)",
    "nRMSE (%)",
]

display_results[numeric_columns] = (
    display_results[numeric_columns].round(4)
)

print(f"\n随机种子 {RANDOM_STATE} 的所有模型测试集评价结果：\n")
print(display_results.to_string(index=False))


# =========================
# 保存评价结果和所有模型预测值
# =========================
if SAVE_RESULTS:
    metrics_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_output_path = (
        metrics_folder / "metrics_all_models.csv"
    )

    predictions_output_path = (
        seed_folder
        / "predict"
        / "predictions_all_models.csv"
    )

    results_df.to_csv(
        metrics_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_results.to_csv(
        predictions_output_path,
        index=True,
        encoding="utf-8-sig",
    )

    print(f"\n评价指标已保存至：{metrics_output_path}")
    print(f"所有模型预测值已保存至：{predictions_output_path}")
