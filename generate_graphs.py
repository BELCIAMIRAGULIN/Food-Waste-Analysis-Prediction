
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Load dataset
data_path = BASE_DIR / "cleaned_food_waste.csv"

if not data_path.exists():
    data_path = BASE_DIR / "train.csv"

df = pd.read_csv(data_path)

# Find target column
target = "food_waste_kg"

if target not in df.columns:
    raise ValueError("Column food_waste_kg not found in dataset")

# Find available columns
def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

meals = find_column([
    "meals_served", "number_of_meals_served"
])

category = find_column([
    "food_category", "food_type", "category"
])

experience = find_column([
    "staff_experience", "staff_experience_years"
])

# 1. Meals vs Food Waste
if meals:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[meals], df[target], alpha=0.6)
    plt.xlabel("Meals Served")
    plt.ylabel("Food Waste (kg)")
    plt.title("Meals Served vs Food Waste")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "meals_vs_waste.png")
    plt.close()

# 2. Waste by Category
if category:
    df.groupby(category)[target].mean().plot(
        kind="bar", figsize=(8, 5)
    )
    plt.title("Average Waste by Category")
    plt.ylabel("Food Waste (kg)")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "waste_by_category.png")
    plt.close()

# 3. Waste by Experience
if experience:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[experience], df[target], alpha=0.6)
    plt.xlabel("Staff Experience")
    plt.ylabel("Food Waste (kg)")
    plt.title("Staff Experience vs Food Waste")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "waste_by_experience.png")
    plt.close()

# 4. Correlation Heatmap
import seaborn as sns

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "correlation_heatmap.png")
    plt.close()

# 5. Outlier Detection
plt.figure(figsize=(7, 5))
plt.boxplot(df[target].dropna())
plt.ylabel("Food Waste (kg)")
plt.title("Food Waste Outlier Detection")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "outlier_detection.png")
plt.close()

print("Graph generation completed!")
print("Available graph files:")

for file in GRAPH_DIR.glob("*.png"):
    print(file.name)

if not meals:
    print("Skipped meals graph: meals column not found")

if not category:
    print("Skipped category graph: category column not found")

if not experience:
    print("Skipped experience graph: experience column not found")
