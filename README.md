# SFRC Strength Prediction

This repository provides an AutoML workflow for predicting the compressive strength of steel fiber-reinforced concrete (SFRC).

## Usage

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib openpyxl autogluon.tabular shap jupyter`
2. The dataset is stored in `data/dataset.xlsx` and the target column is `fc/MPa`.
3. Run `python 100-loop_data.py` to generate train/test splits for 200 random seeds.
4. Run `python 101-loop_train.py` to train AutoGluon models.
5. Run `python 102-loop_predict.py` to generate predictions.
6. Run `python 104_metrics.py` and `python 105-monte_carlo_summary.py` to evaluate and summarize model performance.
7. Open `106-SFRC_shap.ipynb` for SHAP-based model interpretation.

Pretrained models and example outputs are available in the `models/` directory. Run all commands from the repository root; the full 200-seed workflow may take considerable time.
