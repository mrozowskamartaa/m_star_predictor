from dataclasses import dataclass, asdict
import json
from typing import Optional, Tuple, Union, Literal
import os
from dataclasses import asdict

from m_star_predictor import FCNN, FCNN_corrected

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


def recover_M_from_dM(
        dM: np.ndarray,
        n_time_steps: int,
        init_val: int = 0
) -> np.ndarray:
    M = np.nancumsum(dM.reshape(-1, n_time_steps), axis=1)
    return np.pad(M, pad_width=((0,0),(1,0)), mode="constant", constant_values=init_val).flatten()


def normalize(
        data: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
) -> torch.Tensor:
    return (data - mean) / std


def real_units(
        normalized_data: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
) -> torch.Tensor:
    return normalized_data*std + mean


def reduce_dataset(
        X_data: torch.Tensor,
        y_data: torch.Tensor,
        random_state: int,
        fraction: float = 0.1,
) -> torch.Tensor:
    _, X, _, y = train_test_split(
        X_data, y_data, test_size=fraction, random_state=random_state
    )
    return X, y


def compute_rmse(
        y_pred: torch.Tensor,
        y_true: torch.Tensor
) -> torch.Tensor:
    return ((y_pred-y_true)**2).mean().sqrt()


@dataclass
class MStarPredictorConfig:
    dataset_name: str
    dataset_filename: str
    validation_dataset_name: str
    validation_dataset_filename: str
    weights_save_path: str
    feature_names: list[str]
    feature_transforms: Optional[dict[str,Literal["sqrt", "log10"]]]
    target_name: str
    target_transform: Optional[Literal["sqrt", "log10"]]
    network: Literal["fcnn", "fcnn_corrected"]
    n_neurons: Optional[int]
    n_hidden_layers: Optional[int]
    n_neurons_list: Optional[list[int]]
    learning_rate: float
    weight_decay: float
    activation: Literal["relu", "tanh"]
    loss: str
    n_epochs: int
    random_state: int
    train_batch_size: int
    test_batch_size: int
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: list[float]
    target_std: list[float]
    train_loss_trajectory: Optional[list[float]]
    test_loss_trajectory: Optional[list[float]]
    n_time_steps: Union[list[int], int]
    n_time_steps_val: Union[list[int], int]
    rmse: float


@dataclass
class MStarAutoregressivePredictorConfig:
    dataset_name: str
    dataset_filename: str
    validation_dataset_name: str
    validation_dataset_filename: str
    weights_save_path: str
    feature_names: list[str]
    feature_transforms: Optional[dict[str,Literal["sqrt", "log10"]]]
    target_name: str
    target_transform: Optional[Literal["sqrt", "log10"]]
    network: Literal["fcnn", "fcnn_corrected"]
    n_neurons: Optional[int]
    n_hidden_layers: Optional[int]
    n_neurons_list: Optional[list[int]]
    learning_rate: float
    weight_decay: float
    activation: Literal["relu", "tanh"]
    loss: str
    n_epochs: int
    random_state: int
    train_batch_size: int
    test_batch_size: int
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: list[float]
    target_std: list[float]
    train_loss_trajectory: Optional[list[float]]
    test_loss_trajectory: Optional[list[float]]
    n_time_steps: Union[list[int], int]
    n_time_steps_val: Union[list[int], int]
    rmse: float
    k: int


def save_config_to_json(
        config: MStarPredictorConfig,
        predictor_dir: str
) -> None:
    json_string = json.dumps(asdict(config))
    json_save_path = os.path.join(predictor_dir, "config.json")

    with open(json_save_path, "w") as f:
        json.dump(asdict(config), f)


def prepare_data(
        dataframe: pd.DataFrame,
        feature_names: list[str],
        feature_transforms: Optional[dict[str,str]],
        target_name: str,
        target_transform: Optional[str],
        autoregressive: bool = False,
        n_timesteps: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    X_transformed = transform_features(
        dataframe=dataframe, feature_names=feature_names, feature_transforms=feature_transforms
    )

    y_transformed = transform_target(
        dataframe=dataframe, target_name=target_name, target_transform=target_transform
    )

    X = torch.tensor(np.array(X_transformed)).T.float()
    y = torch.tensor(y_transformed).reshape(-1,1).float()

    if autoregressive:
        assert n_timesteps is not None, "n_timesteps required for autoregressive datasets"
        X = X.reshape(-1, n_timesteps, len(feature_names))
        y = y.reshape(-1, n_timesteps, 1)

    return X, y


def transform_features(
        dataframe: pd.DataFrame,
        feature_names: list[str],
        feature_transforms: Optional[dict[str,str]]
):

    X_transformed = []

    for feature_name in feature_names:
        if feature_transforms and feature_name in feature_transforms:
            transform = feature_transforms[feature_name]
            if transform == "sqrt":
                X_transformed.append(np.sqrt(dataframe[feature_name].values))
            elif transform == "log10":
                X_transformed.append(np.log10(dataframe[feature_name].values))
            elif transform == "log":
                X_transformed.append(np.log(1+dataframe[feature_name].values))
        else:
            X_transformed.append(dataframe[feature_name].values)

    return X_transformed


def transform_target(
        dataframe: pd.DataFrame,
        target_name: str,
        target_transform: Optional[str]
):
    if target_transform == "sqrt":
        return np.sqrt(dataframe[target_name].values)
    elif target_transform == "log10":
        return np.log10(dataframe[target_name].values)
    elif target_transform == "log":
        return np.log(1+dataframe[target_name].values)
    else:
        return dataframe[target_name].values


def get_predictions(
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        config: MStarPredictorConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if config.activation == "relu":
        activation = torch.nn.ReLU()
    elif config.activation == "tanh":
        activation = torch.nn.Tanh()
    else:
        raise ValueError("Unsupported activation function (try 'relu' or 'tanh').")

    X, y = prepare_data(
        dataframe=training_data, feature_names=config.feature_names, feature_transforms=config.feature_transforms,
        target_name=config.target_name, target_transform=None
    )

    X_val, y_val = prepare_data(
        dataframe=validation_data, feature_names=config.feature_names, feature_transforms=config.feature_transforms,
        target_name=config.target_name, target_transform=None
    )

    # TODO: rn hard coded FCNN but Linear and simple_FCNN should also be supported
    if config.network == "fcnn":
        network = FCNN(
            input_size=len(config.feature_names), n_neurons=config.n_neurons, n_hidden_layers=config.n_hidden_layers, 
            activation=activation)
    elif config.network == "fcnn_corrected":
        network = FCNN_corrected(
            input_size=len(config.feature_names), n_neurons_list=config.n_neurons_list, activation=activation)

    state_dict = torch.load(config.weights_save_path, map_location=torch.device('cpu'))
    network.load_state_dict(state_dict)

    X_mean = torch.tensor(config.feature_mean)
    X_std = torch.tensor(config.feature_std)
    y_mean = torch.tensor(config.target_mean)
    y_std = torch.tensor(config.target_std)

    X_normalized = normalize(data=X, mean=X_mean, std=X_std)
    fcnn_prediction = real_units(normalized_data=network(X_normalized), mean=y_mean, std=y_std)

    X_val_normalized = normalize(data=X_val, mean=X_mean, std=X_std)
    fcnn_prediction_val = real_units(normalized_data=network(X_val_normalized), mean=y_mean, std=y_std)

    if config.target_transform == "sqrt":
        fcnn_prediction = fcnn_prediction**2
        fcnn_prediction_val = fcnn_prediction_val**2
    elif config.target_transform == "log10":
        fcnn_prediction = 10**fcnn_prediction
        fcnn_prediction_val = 10**fcnn_prediction_val
    elif config.target_transform == "log":
        fcnn_prediction = np.exp(fcnn_prediction)-1
        fcnn_prediction_val = np.exp(fcnn_prediction_val)-1
    else:
        pass

    return X, y, X_val, y_val, fcnn_prediction, fcnn_prediction_val


def get_mean_predictions(
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        config: MStarPredictorConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    mean_feature_names = ["f", "u_star", "bl_rh18", "B"]
    path_to_weights = "/home/mm6851/m_star_predictor/m_star_predictors/fcnn_mean_predictor_weights.pth"

    data_directory = '../m_star_dataset'
    dataset_name = 'ePBL_paper_expanded_2283_corrected'

    mean_dataset_filename = f"{dataset_name}_mean_dataset_M_target.pt"
    mean_data = torch.load(os.path.join(data_directory, mean_dataset_filename))

    X, y = prepare_data(
        dataframe=training_data, feature_names=mean_feature_names, feature_transforms={"": ""},
        target_name=["M"], target_transform=config.target_transform
    )

    X_val, y_val = prepare_data(
        dataframe=validation_data, feature_names=mean_feature_names, feature_transforms={"": ""},
        target_name=["M"], target_transform=config.target_transform
    )

    # TODO: rn hard coded FCNN but Linear and simple_FCNN should also be supported
    network = FCNN(
        input_size=len(mean_feature_names), n_neurons=16, n_hidden_layers=1, 
        activation=torch.nn.ReLU())

    state_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
    network.load_state_dict(state_dict)

    X_m, y_m = mean_data.T[:,:4].float(), mean_data.T[:,-1].unsqueeze(1).float()

    X_mp_mean = X_m.mean(dim=0)
    X_mp_std = X_m.std(dim=0)
    y_mp_mean = y_m.mean(dim=0)
    y_mp_std = y_m.std(dim=0)

    X_normalized = normalize(data=X, mean=X_mp_mean, std=X_mp_std)
    mean_prediction = real_units(normalized_data=network(X_normalized), mean=y_mp_mean, std=y_mp_std)

    X_val_normalized = normalize(data=X_val, mean=X_mp_mean, std=X_mp_std)
    mean_prediction_val = real_units(normalized_data=network(X_val_normalized), mean=y_mp_mean, std=y_mp_std)

    return mean_prediction, mean_prediction_val
