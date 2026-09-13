lable_str = "fc/MPa"
# random_state = 82

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# read data
data_raw = pd.read_excel("data/dataset.xlsx")

for random_state in range(200):
    # split data
    X = data_raw.drop(lable_str, axis=1)
    y = data_raw[lable_str]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)


    # Save data
    save_folder = Path(f"output/seed_{random_state}")
    (save_folder/"data").mkdir(exist_ok=True, parents=True)

    X.to_csv(save_folder/"data"/"X.csv")
    y.to_csv(save_folder/"data"/"y.csv")
    train_data.to_csv(save_folder/"data"/"train_data.csv")
    test_data.to_csv(save_folder/"data"/"test_data.csv")
    X_train.to_csv(save_folder/"data"/"X_train.csv")
    y_train.to_csv(save_folder/"data"/"y_train.csv")
    X_test.to_csv(save_folder/"data"/"X_test.csv")
    y_test.to_csv(save_folder/"data"/"y_test.csv")
