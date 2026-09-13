lable_str = "fc/MPa"
random_state = 82

from pathlib import Path
import pandas as pd
from autogluon.tabular import TabularPredictor


for random_state in range(200):
     ## load data
     save_folder = f"output/seed_{random_state}"
     train_data = pd.read_csv(save_folder+"/data"+"/train_data.csv", index_col=0)
     
     ## train model
     predictor = TabularPredictor(label=lable_str, eval_metric="r2", path=save_folder+"/ag_model")
     predictor.fit(train_data)
