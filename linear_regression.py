import numpy as np
import pandas as pd
import sklearn.linear_model as LinearModel
import sklearn.metrics as Metrics
import sklearn.model_selection as ModelSelection
from sklearn import tree
import xgboost_trial as xgb

train_data_file = "song-dataset/train.csv"
test_data_file = "song-dataset/test.csv"

def load_data(train_data_file, test_data_file):
    train_data = pd.read_csv(train_data_file)
    train_id = train_data["id"]
    test_data = pd.read_csv(test_data_file)
    test_id = test_data["id"]
    return train_data, test_data, train_id, test_id

train_data, test_data, train_id, test_id = load_data(train_data_file, test_data_file)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
train_data = train_data.drop(columns=["id"])
test_data = test_data.drop(columns=["id"])

model = LinearModel.LinearRegression()
model.fit(train_data.drop(columns=["song_popularity"]), train_data["song_popularity"])

predictions = model.predict(test_data)

new_data = pd.DataFrame({"id": test_id, "song_popularity": predictions})
new_data.to_csv("song-dataset/lr.csv", index=False)
