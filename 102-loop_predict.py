lable_str = "fc/MPa"
random_state = 82

from pathlib import Path
from autogluon.tabular import TabularPredictor
import pandas as pd

for random_state in range(200):
    ## load data
    save_folder = Path(f"output/seed_{random_state}")
    test_data = pd.read_csv(save_folder / "data" / "test_data.csv", index_col=0)
    X_test = pd.read_csv(save_folder / "data" / "X_test.csv", index_col=0)

    train_data = pd.read_csv(save_folder / "data" / "train_data.csv", index_col=0)
    X_train = pd.read_csv(save_folder / "data" / "X_train.csv", index_col=0)

    ## load model
    predictor = TabularPredictor(label=lable_str).load(save_folder / "ag_model")

    ## predict
    y_pred = predictor.predict(X_test)
    x_pred = predictor.predict(X_train)

    ## save data
    Path(save_folder / "predict").mkdir(exist_ok=True, parents=True)
    y_pred.to_csv(save_folder / "predict" / "y_pred.csv")

    Path(save_folder / "predict").mkdir(exist_ok=True, parents=True)
    x_pred.to_csv(save_folder / "predict" / "x_pred.csv")
