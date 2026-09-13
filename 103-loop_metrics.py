from pandas import DataFrame

lable_str = "fc/MPa"
random_state = 82
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


for random_state in range(200):
    save_folder = Path(f"output/seed_{random_state}")

    y_test = pd.read_csv(save_folder/"data"/"y_test.csv", index_col=0)
    y_pred: DataFrame = pd.read_csv(save_folder/"predict"/"y_pred.csv", index_col=0)


    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(r2, rmse, mae,mape)
