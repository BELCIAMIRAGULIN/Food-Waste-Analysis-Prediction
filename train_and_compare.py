
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

root = Path(__file__).resolve().parent

for folder in ["graphs", "models", "results"]:
    (root / folder).mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(root / "train.csv")

X = df.drop(columns=["food_waste_kg", "ID"], errors="ignore")
y = df["food_waste_kg"]

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "R2": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    })

    if name == "Random Forest":
        joblib.dump(pipeline, root / "models" / "random_forest.pkl")

results_df = pd.DataFrame(results)
results_df.to_csv(
    root / "results" / "model_comparison.csv",
    index=False
)

# Generate model comparison graphs
for metric, filename in [
    ("R2", "r2_comparison.png"),
    ("RMSE", "rmse_comparison.png")
]:
    results_df.plot(
        x="Model",
        y=metric,
        kind="bar",
        legend=False,
        title=f"{metric} Model Comparison"
    )

    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(root / "graphs" / filename)
    plt.close()

print("Model and graphs generated successfully!")
