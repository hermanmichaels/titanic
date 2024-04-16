import matplotlib.pyplot as plt
from data import extract_title
import numpy as np
import pandas as pd


def plot_survival_ratio(df: pd.DataFrame, column: str) -> None:
    """Plot the survival ratio for a discrete feature - i.e. the survival ratio
    per unique feature value.

    Args:
        df: dataframe containing the data
        column: column name
    """
    # Replace NaN tokens by "Unk(nown)"
    df[column].fillna("Unk", inplace=True)

    # Calculate survival ratio per existing value in the column
    total_counts = df[column].value_counts()
    survival_counts = (
        df.groupby([column, "Survived"]).size().unstack().fillna(0)
    )
    column_values = df[column].unique()
    survival_ratios = [
        survival_counts.loc[value, 1] / total_counts[value]
        for value in column_values
    ]

    color_mapping = plt.cm.get_cmap("tab10", len(survival_counts))
    bar_colors = color_mapping(np.arange(len(column_values)))

    # Plot the survival ratios
    plt.bar(column_values, survival_ratios, color=bar_colors)
    plt.title(f"Survival Ratio by {column}")
    plt.xlabel(column)
    plt.ylabel("Survival Ratio")
    plt.ylim(0, 1)
    plt.show()


def plot_survival_cont(df: pd.DataFrame, column: str) -> None:
    """Plot the survival ratio for a continous feature - for this
    we bin the feature values first.

    Args:
        df: dataframe containing the data
        column: column name
    """
    # Drop NaN values
    df[column].dropna(inplace=True)

    # Bin values into 10 categories
    num_bins = 10
    bin_edges = np.linspace(min(df[column]), max(df[column]), num_bins)
    bin_centers = [
        (bin_edges[i] + bin_edges[i + 1]) / 2
        for i in range(len(bin_edges) - 1)
    ]
    bin_width = bin_edges[1] - bin_edges[0]

    df[f"{column}_bin"] = pd.cut(
        df[column], bins=bin_edges, include_lowest=True, right=True
    )

    # Calculate the ratio of 'Survived' per 'column' bin
    ratio_survived_per_bin = df.groupby(f"{column}_bin")["Survived"].mean()

    plt.bar(
        bin_centers,
        ratio_survived_per_bin.values,
        width=bin_width * 0.8,
    )
    plt.xlabel(column)
    plt.ylabel("Ratio of Survived")
    plt.title(f"Ratio of Survived per {column} Bin")
    plt.show()

def main():
    df = pd.read_csv("titanic/train.csv")

    plot_survival_cont(df, "Age")
    plot_survival_cont(df, "Fare")

    extract_title(df)
    plot_survival_ratio(df, "Title")

    family_size = df["Parch"] + df["SibSp"]
    df["Alone"] = family_size == 0
    plot_survival_ratio(df, "Alone")

    plot_survival_ratio(df, "Sex")
    plot_survival_ratio(df, "Pclass")
    plot_survival_ratio(df, "SibSp")
    plot_survival_ratio(df, "Parch")
    plot_survival_ratio(df, "Embarked")

if __name__ == "__main__":
    main()
