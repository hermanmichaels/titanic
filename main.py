import math

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from torch.optim.lr_scheduler  import MultiStepLR

from data import prepare_data, prepare_test_data, prepare_train_data
from model import MLP
from utils import set_seeds

PATH_TO_TRAIN_FILE = "titanic/train.csv"
PATH_TO_TEST_FILE = "titanic/test.csv"
INPUT_SIZE = 18
OUTPUT_SIZE = 2
NUM_EPOCHS = 100
LR = 0.001
RANDOM_SEED = 42
MODEL_PATH = "/tmp/model.ckpt"


def train() -> tuple[ColumnTransformer, torch.nn.Module]:
    """Main training loop.

    Returns:
        tuple of fit transformation pipeline and trained model
    """
    df = prepare_data(PATH_TO_TRAIN_FILE)

    (
        train_loader,
        val_loader,
        pipeline,
    ) = prepare_train_data(df)

    model = MLP(INPUT_SIZE, OUTPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[80], gamma=0.1)

    best_val_acc = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits, predidctions = model(inputs)

            loss = criterion(
                logits,
                torch.nn.functional.one_hot(
                    labels.long(), num_classes=2
                ).float(),
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            correct += (predidctions == labels).float().mean().item()

        scheduler.step()

        loss = running_loss / len(train_loader)
        acc = correct / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss:.4f}, Accuracy: {acc:.4f}"
        )

        if epoch % 10 == 9:
            # Evaluation on the val set
            model.eval()
            with torch.no_grad():
                correct = 0
                for inputs, labels in val_loader:
                    _, predictions = model(inputs)
                    correct += (predictions == labels).float().mean().item()

                acc = correct / len(val_loader)
                print(f"Accuracy on val set: {acc:.4f}")

                if acc > best_val_acc:
                    best_val_acc = acc
                    print("New best model found")
                    torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH))

    return pipeline, model


def main():
    # Make experiments reproducible
    set_seeds(RANDOM_SEED)

    # Train the model
    pipeline, model = train()

    # Load the test dataset for making predictions
    df = prepare_data(PATH_TO_TEST_FILE)
    test_loader, ids = prepare_test_data(df, pipeline)

    # Make predictions on the test set
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            _, predidctions = model(inputs)
            test_predictions.extend(predidctions.numpy())

    output_df = pd.DataFrame(
        {
            "PassengerId": ids,
            "Survived": [
                int(x) if not math.isnan(x) else 0 for x in test_predictions
            ],
        }
    )
    output_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
