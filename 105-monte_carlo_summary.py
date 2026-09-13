from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("output")
SUMMARY_DIR = METRICS_DIR / "monte_carlo_summary"
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

METRIC_COLUMNS = ["r2", "rmse", "mae", "mape", "mse", "nrmse"]
MODEL_LABELS = {
    "WeightedEnsemble_L2": "AutoML",
}


def seed_number(path: Path) -> int:
    return int(path.parent.parent.name.split("_")[-1])


def load_metric_files() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(METRICS_DIR.glob("seed_*/metrics/test_metrics.csv"), key=seed_number):
        df = pd.read_csv(csv_path)
        if "model_names" not in df.columns:
            raise ValueError(f"Missing model_names column in {csv_path}")

        df = df.rename(columns={"model_names": "model"})
        df["seed"] = seed_number(csv_path)

        for metric in METRIC_COLUMNS:
            if metric not in df.columns:
                df[metric] = pd.NA
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        y_test_path = csv_path.parent.parent / "data" / "y_test.csv"
        if y_test_path.exists() and df["rmse"].notna().any():
            y_test = pd.read_csv(y_test_path, index_col=0).squeeze("columns")
            y_mean = y_test.mean()
            if pd.notna(y_mean) and y_mean != 0:
                df["nrmse"] = df["nrmse"].fillna(df["rmse"] / y_mean * 100)

        frames.append(df[["seed", "model", *METRIC_COLUMNS]])

    if not frames:
        raise FileNotFoundError("No output/seed_*/metrics/test_metrics.csv files found.")

    return pd.concat(frames, ignore_index=True)


def format_range(series: pd.Series, digits: int = 3) -> str:
    valid = series.dropna()
    if valid.empty:
        return ""
    return f"{valid.min():.{digits}f}-{valid.max():.{digits}f}"


def format_mean_std(series: pd.Series, digits: int = 3) -> str:
    valid = series.dropna()
    if valid.empty:
        return ""
    if len(valid) == 1:
        return f"{valid.mean():.{digits}f}"
    return f"{valid.mean():.{digits}f} +/- {valid.std(ddof=1):.{digits}f}"


def format_publication_mean_std(series: pd.Series, digits: int = 4) -> str:
    valid = series.dropna()
    if valid.empty:
        return ""
    if len(valid) == 1:
        return f"{valid.mean():.{digits}f}"
    return f"{valid.mean():.{digits}f} (±{valid.std(ddof=1):.{digits}f})"


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw["rmse"].notna().any():
        raw = raw.copy()
        raw["mse"] = raw["mse"].fillna(raw["rmse"] ** 2)

    rows = []
    for model, group in raw.groupby("model", sort=False):
        row = {
            "Model": MODEL_LABELS.get(model, model),
            "Seeds": group["seed"].nunique(),
            "_r2_mean": group["r2"].mean(),
            "R2 range": format_range(group["r2"]),
            "R2 mean +/- SD": format_mean_std(group["r2"]),
            "RMSE range": format_range(group["rmse"]),
            "RMSE mean +/- SD": format_mean_std(group["rmse"]),
            "MAE range": format_range(group["mae"]),
            "MAE mean +/- SD": format_mean_std(group["mae"]),
            "nRMSE range": format_range(group["nrmse"]),
            "nRMSE mean +/- SD": format_mean_std(group["nrmse"]),
        }

        if group["mape"].notna().any():
            row["MAPE range"] = format_range(group["mape"])
            row["MAPE mean +/- SD"] = format_mean_std(group["mape"])

        if group["mse"].notna().any():
            row["MSE range"] = format_range(group["mse"])
            row["MSE mean +/- SD"] = format_mean_std(group["mse"])

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("_r2_mean", ascending=False)
    summary = summary.drop(columns="_r2_mean")

    return summary.reset_index(drop=True)


def build_publication_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw["rmse"].notna().any():
        raw = raw.copy()
        raw["mse"] = raw["mse"].fillna(raw["rmse"] ** 2)

    rows = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "Label": MODEL_LABELS.get(model, model),
                "_r2_mean": group["r2"].mean(),
                "R2": format_publication_mean_std(group["r2"]),
                "RMSE": format_publication_mean_std(group["rmse"]),
                "MAE": format_publication_mean_std(group["mae"]),
                "MSE": format_publication_mean_std(group["mse"]),
                "nRMSE": format_publication_mean_std(group["nrmse"]),
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values("_r2_mean", ascending=False).drop(columns="_r2_mean")
    return table.reset_index(drop=True)


def save_table_image(summary: pd.DataFrame, output_path: Path) -> None:
    image_df = summary.copy()
    columns = list(image_df.columns)
    rows = len(image_df)
    fig_width = max(13, len(columns) * 1.55)
    fig_height = max(5, rows * 0.42 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.axis("off")

    table = ax.table(
        cellText=image_df.values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
        else:
            cell.set_facecolor("white")

    for col in range(len(columns)):
        table.auto_set_column_width(col)

    ax.set_title("Monte Carlo Validation Metric Ranges by Model", fontsize=12, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_publication_table_image(table_df: pd.DataFrame, output_path: Path) -> None:
    rows = len(table_df)
    fig_width = 12.8
    fig_height = max(3.4, 0.34 * rows + 1.15)

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis("off")

    ax.text(
        0.5,
        0.98,
        "Table 4. Robustness evaluation under Monte Carlo cross-validation.",
        ha="center",
        va="top",
        fontsize=10.5,
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.25, 0.16, 0.16, 0.16, 0.16, 0.16],
        bbox=[0.02, 0.04, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.16)

    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        if row == 0:
            cell.get_text().set_weight("bold")

    ncols = len(table_df.columns)
    nrows = len(table_df) + 1

    for col in range(ncols):
        header_cell = table[(0, col)]
        header_cell.visible_edges = "TB"
        header_cell.set_linewidth(0.9)

        bottom_cell = table[(nrows - 1, col)]
        bottom_cell.visible_edges = "B"
        bottom_cell.set_linewidth(0.9)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    raw = load_metric_files()
    summary = build_summary(raw)
    publication_table = build_publication_table(raw)

    raw_path = SUMMARY_DIR / "monte_carlo_metrics_raw.csv"
    csv_path = SUMMARY_DIR / "monte_carlo_metric_ranges.csv"
    xlsx_path = SUMMARY_DIR / "monte_carlo_metric_ranges.xlsx"
    png_path = SUMMARY_DIR / "monte_carlo_metric_ranges.png"
    publication_csv_path = SUMMARY_DIR / "monte_carlo_publication_table.csv"
    publication_xlsx_path = SUMMARY_DIR / "monte_carlo_publication_table.xlsx"
    publication_png_path = SUMMARY_DIR / "monte_carlo_publication_table.png"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(csv_path, index=False)
    summary.to_excel(xlsx_path, index=False)
    save_table_image(summary, png_path)
    publication_table.to_csv(publication_csv_path, index=False)
    publication_table.to_excel(publication_xlsx_path, index=False)
    save_publication_table_image(publication_table, publication_png_path)

    print(f"Loaded {raw['seed'].nunique()} seeds and {raw['model'].nunique()} models.")
    print(f"Raw metrics: {raw_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary XLSX: {xlsx_path}")
    print(f"Summary PNG: {png_path}")
    print(f"Publication CSV: {publication_csv_path}")
    print(f"Publication XLSX: {publication_xlsx_path}")
    print(f"Publication PNG: {publication_png_path}")


if __name__ == "__main__":
    main()
