import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """Simple MLP model with which we will fit the Titanic dataset."""
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Model forward.

        Args:
            x: input tensor [bs, input_size]

        Returns:
            tuple of output value before sigmoid [bs, 2] and resulting prediction [bs]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        pred = self.sigmoid(x)
        return x, torch.argmax(pred, dim=1)
