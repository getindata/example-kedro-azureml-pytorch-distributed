import logging
from typing import Dict, Tuple, Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kedro_azureml.distributed import distributed_job
from kedro_azureml.distributed.config import Framework
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import TensorDataset, DataLoader


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def create_dataloader(
    x: pd.DataFrame, y: pd.Series = None, predict=False, batch_size=256
):
    data = [torch.from_numpy(x.values).float()]
    if y is not None:
        data.append(torch.from_numpy(y.values).float())
    return DataLoader(TensorDataset(*data), shuffle=not predict, batch_size=batch_size)


@distributed_job(Framework.PyTorch, num_nodes="params:model_options.num_nodes")
def train_model_pytorch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_nodes: int,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
):
    class SimpleNetwork(pl.LightningModule):
        def __init__(self, n_features: int, lr: float) -> None:
            super().__init__()
            self.lr = lr
            self.normalize = nn.BatchNorm1d(n_features)
            internal_features = 1024
            hidden_layer_size = 128
            depth = 10
            self.layers = nn.Sequential(
                nn.Linear(n_features, internal_features),
                nn.Sequential(
                    nn.Linear(internal_features, hidden_layer_size),
                    nn.LeakyReLU(),
                    *sum(
                        [
                            [
                                nn.Linear(hidden_layer_size, hidden_layer_size),
                                nn.LeakyReLU(),
                            ]
                            for _ in range(depth)
                        ],
                        [],
                    ),
                ),
                nn.Linear(hidden_layer_size, 1, bias=False),
            )

        def forward(self, x):
            normalized = self.normalize(x)
            outputs = self.layers(normalized)
            return outputs.squeeze()

        def training_step(self, batch, batch_idx):
            x, y = batch
            outputs = self.forward(x)
            loss = F.smooth_l1_loss(outputs, y)
            return loss

        def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: int = 0
        ) -> Any:
            return self.forward(batch[0])

        def configure_optimizers(self):
            return Adagrad(self.parameters(), lr=self.lr)

    epochs = max_epochs
    data = create_dataloader(X_train.astype("float"), y_train, batch_size=batch_size)
    model = SimpleNetwork(X_train.shape[1], learning_rate)

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=True,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        num_nodes=num_nodes,
        accelerator="auto",
    )

    trainer.fit(model, train_dataloaders=data)
    return model


def evaluate_model(model: pl.LightningModule, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        model: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """

    with torch.no_grad():
        trainer = pl.Trainer()
        dataloader = create_dataloader(X_test.astype("float"), predict=True)
        y_pred = trainer.predict(model, dataloaders=dataloader)
        y_pred = pd.Series(
            index=y_test.index, data=torch.cat(y_pred).reshape(-1).numpy()
        )

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", r2)
    logger.info("Model has MAE of %.3f on test data.", mae)
