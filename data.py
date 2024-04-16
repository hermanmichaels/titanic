import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

NUMERIC_COLUMNS = ["Age", "Fare"]
CATEGORICAL_COLUMNS = ["Sex", "Pclass", "Embarked", "Title", "Alone"]

TARGET_COLUMN = "Survived"
BS = 64
RANDOM_SEED = 42
TEST_RATIO = 0.2


class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def extract_title(df: pd.DataFrame) -> None:
    """Extracts and adds title information / column to data frame,

    Args:
        df: data frame to add "Title" to
    """
    # Match any consecutive string ending with ".",
    # but only grab first part without "."
    df["Title"] = df.Name.str.extract(r"([A-Za-z]+)\.", expand=False)

    # Map other titles / spellings to some fixed "common titles",
    # and map others to "Others"
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"].replace(["Ms", "Mlle", "Mme"], "Miss", inplace=True)
    df["Title"].replace(["Lady"], "Mrs", inplace=True)
    df["Title"].replace(["Sir", "Rev"], "Mr", inplace=True)
    df["Title"][~df.Title.isin(common_titles)] = "Others"


def prepare_data(path: str) -> pd.DataFrame:
    """Prepares data, i.e. reads data file
    and generates all required features.

    Args:
        path: path to data file

    Returns:
        resulting data frame
    """
    df = pd.read_csv(path)

    # Replace NaN / Unk tokens
    df["Age"].fillna(0, inplace=True)
    df["Embarked"].fillna("Unk", inplace=True)

    # Add "Title" column
    extract_title(df)

    # Derive whether passenger is travelling alone
    family_size = df["Parch"] + df["SibSp"]
    df["Alone"] = family_size == 0

    return df


def generate_data_loader(
    x: np.ndarray, y: np.ndarray | None = None, shuffle: bool = True
):
    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = (
        torch.tensor(y, dtype=torch.float32)
        if y is not None
        else torch.zeros_like(X_tensor)
    )

    dataset = TitanicDataset(
        X_tensor,
        y_tensor,
    )
    loader = DataLoader(dataset, batch_size=BS, shuffle=shuffle)

    return loader


def prepare_train_data(
    df: pd.DataFrame,
) -> tuple[DataLoader, DataLoader, ColumnTransformer]:
    """Gets train data ready for the model.
    Extracts column of relevance from the dataset,
    splits into train and val, then transforms
    the features via scikit functionality.
    Lastly generates data loaders for train and val.

    Args:
        df: full dataset

    Returns:
        tuple of: data loaders for train and val, fitted feature transformation pipeline
    """
    x = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df[TARGET_COLUMN].values

    # Split dataset into train and test
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    # Data pre-processing pipeline - fit with train
    # then apply to test
    pipeline = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    x_train = pipeline.fit_transform(x_train)
    x_val = pipeline.transform(x_val)

    return (
        generate_data_loader(x_train, y_train, True),
        generate_data_loader(x_val, y_val, False),
        pipeline,
    )


def prepare_test_data(
    df: pd.DataFrame, pipeline: ColumnTransformer
) -> tuple[DataLoader, pd.DataFrame]:
    """Similar to prepare_train_data, gets test data ready
    for the model.
    Transforms features via previously fit pipeline,
    generate data loader and additionally return passenger ids
    for the submission.

    Args:
        df: test dataset
        pipeline: pipeline fit on train data

    Returns:
        _uple of test data loader and passenger ids
    """
    x = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    x = pipeline.transform(x)
    return generate_data_loader(x, None, False), df["PassengerId"]
