import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    filename='song-dataset/kfold-ensemble/xgboost_kfold-7.log',
    level=logging.INFO,           # log level
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'                  # 'w' to overwrite, 'a' to append
)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:
            self.level(message)

    def flush(self):
        pass

sys.stdout = LoggerWriter(logging.info)  # redirect print to logging
sys.stderr = LoggerWriter(logging.error)


# ---------------------------
# Load data
# ---------------------------
train_file = "song-dataset/train.csv"
test_file = "song-dataset/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

train_df = train_df.fillna(train_df.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))

train_id = train_df["id"].copy()
test_id = test_df["id"].copy()
train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])

X = train_df.drop(columns=["song_popularity"])
y = train_df["song_popularity"]

# ---------------------------
# K-Fold CV using xgb.train
# ---------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))

for seed in range(1, 26):
    final_preds = np.zeros(len(test_df))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"Training fold {fold}/5...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(test_df)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.01,
            "max_depth": 5,
            "subsample": 0.75,
            "colsample_bytree": 0.8,
            "seed": seed,
            "eval_metric": "rmse"
        }

        evals = [(dtrain, "train"), (dval, "valid")]

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )

        val_preds = bst.predict(dval)
        oof_preds[val_idx] += val_preds/25  # average over seeds
        fold_test_preds = bst.predict(dtest)
        final_preds += fold_test_preds / kf.n_splits
    test_preds += final_preds / 25  # average over seeds

# CV RMSE
rmse = mean_squared_error(y, oof_preds) ** 0.5
print("XGBoost CV RMSE:", rmse)

# ---------------------------
# RandomForest ensemble
# ---------------------------
print("Training RandomForest on full data...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_preds = rf.predict(test_df)

# Blend XGB + RF
final_preds = 0.7 * test_preds + 0.3 * rf_preds

# ---------------------------
# Save predictions
# ---------------------------
os.makedirs("song-dataset", exist_ok=True)
output_df = pd.DataFrame({"id": test_id, "song_popularity": final_preds})
output_df.to_csv("song-dataset/kfold-ensemble/xgboost-ensemble-7.csv", index=False)
print("Predictions saved to song-dataset/xgboost-ensemble.csv")

# Save hyperparameters
with open("song-dataset/kfold-ensemble/xgboost-ensemble-params-7.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")
    f.write("RandomForestRegressor:\n")
    f.write("n_estimators: 300\n")
    f.write("max_depth: 12\n")
    f.write("random_state: 42\n")
print("Hyperparameters saved to song-dataset/xgboost-ensemble-params.txt")
