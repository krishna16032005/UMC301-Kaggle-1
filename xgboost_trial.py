import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

train_data_file = "song-dataset/train.csv"
test_data_file = "song-dataset/test.csv"

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Load data
train_df, test_df = load_data(train_data_file, test_data_file)

# Fill missing values
train_df = train_df.fillna(train_df.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))

# Save IDs
train_id = train_df["id"]
test_id = test_df["id"]

# Drop ID column
train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])

# Split features and target
X = train_df.drop(columns=["song_popularity"])
y = train_df["song_popularity"]

# Optional: train-test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgbr = xgb.XGBRegressor(
    n_estimators=100,        # number of trees
    learning_rate=0.1,       # step size shrinkage
    max_depth=5,             # max depth of each tree
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # column sampling
    random_state=42
)

# Train model
xgbr.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = xgbr.predict(X_val)

# Train on full dataset
xgbr.fit(X, y)

# Make predictions for test data
predictions = xgbr.predict(test_df)

# Save predictions
output_df = pd.DataFrame({"id": test_id, "song_popularity": predictions})
output_df.to_csv("song-dataset/xgboost-11.csv", index=False)
print("Predictions saved to song-dataset/xgboost.csv")