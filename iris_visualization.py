# ============================================
# Project: Data Visualization using Iris Dataset
# Author: Your Name
# Dataset: iris.csv (Downloaded from Kaggle)
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("iris.csv")

# Display Dataset Information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Set plot style
sns.set_style("whitegrid")

# ===========================================================
# GRAPH 1 : BAR CHART
# ===========================================================

plt.figure(figsize=(7,5))

species_count = df["Species"].value_counts()

species_count.plot(kind="bar")

plt.title("Species Count")
plt.xlabel("Species")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ===========================================================
# GRAPH 2 : HISTOGRAM
# ===========================================================

plt.figure(figsize=(7,5))

plt.hist(df["SepalLengthCm"],
         bins=10,
         edgecolor="black")

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ===========================================================
# GRAPH 3 : SCATTER PLOT
# Petal Length vs Petal Width
# ===========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    s=80
)

plt.title("Petal Length vs Petal Width")

plt.tight_layout()
plt.show()

# ===========================================================
# GRAPH 4 : FEATURE COMPARISON
# Sepal Length vs Sepal Width
# ===========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species",
    s=80
)

plt.title("Sepal Length vs Sepal Width")

plt.tight_layout()
plt.show()

# ===========================================================
# GRAPH 5 : HEATMAP
# ===========================================================

plt.figure(figsize=(8,6))

numeric_df = df.drop(columns=["Species"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

print("\n===================================")
print("Project Executed Successfully!")
print("Total Graphs Displayed: 5")
print("===================================")
