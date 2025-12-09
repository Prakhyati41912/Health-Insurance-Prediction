# ------------------------------------------------
# 1) Import dependencies
# ------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------
# 2) Load dataset
# ------------------------------------------------
df = pd.read_csv("insurance.csv")

# ------------------------------------------------
# 3) Split features and target
# ------------------------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

# ------------------------------------------------
# 4) Categorical + numeric columns
# ------------------------------------------------
cat_cols = ["sex", "smoker", "region"]
num_cols = ["age", "bmi", "children"]

# ------------------------------------------------
# 5) Preprocessing
# ------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# ------------------------------------------------
# 6) Model initialization — tuned XGBoost
# ------------------------------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ------------------------------------------------
# 7) Create Pipeline
# ------------------------------------------------
pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', model)
])

# ------------------------------------------------
# 8) Train-test split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 9) Train model
# ------------------------------------------------
pipe.fit(X_train, y_train)

# ------------------------------------------------
# 10) Make predictions
# ------------------------------------------------
y_pred = pipe.predict(X_test)

# ------------------------------------------------
# 11) Evaluate performance
# ------------------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("✅ XGBoost Performance")
print("---------------------------")
print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# ------------------------------------------------
# 12) Save model
# ------------------------------------------------
joblib.dump(pipe, "insurance_model.pkl")
print("\n✅ XGBoost Model Trained & Saved Successfully!")

# ------------------------------------------------
# 13) SHAP Explainability
# ------------------------------------------------
print("\n✅ Generating SHAP plots...")

# Extract model inside pipeline
model_xgb = pipe.named_steps['model']

# Transform data for SHAP
X_train_preprocessed = pipe.named_steps['preprocess'].transform(X_train)

# Build SHAP explainer
explainer = shap.TreeExplainer(model_xgb)

# SHAP values
shap_values = explainer.shap_values(X_train_preprocessed)

# Summary plot
shap.summary_plot(shap_values, X_train_preprocessed, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

print("✅ SHAP Summary Plot Saved (shap_summary.png)")
