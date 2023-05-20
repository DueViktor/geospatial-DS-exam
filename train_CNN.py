import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

import data_loader as dl
from CNN import CNN

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")


class AGBM_CNN:
    def __init__(
        self,
        use_gpu=True,
        num_dpoints=10,
        exclude_layer=None,
        num_input_channels=15,
        start_time=None,
    ):
        self.channel_map = {
            0: "S2-B2: Blue-10m",
            1: "S2-B3: Green-10m",
            2: "S2-B4: Red-10m",
            3: "S2-B5: VegRed-704nm-20m",
            4: "S2-B6: VegRed-740nm-20m",
            5: "S2-B7: VegRed-780nm-20m",
            6: "S2-B8: NIR-833nm-10m",
            7: "S2-B8A: NarrowNIR-864nm-20m",
            8: "S2-B11: SWIR-1610nm-20m",
            9: "S2-B12: SWIR-2200nm-20m",
            10: "S2-CLP: Clouse_gpuudProb-160m",
            11: "S1-VV-Asc: Cband-10m",
            12: "S1-VH-Asc: Cband-10m",
            13: "S1-VV-Desc: Cband-10m",
            14: "S1-VH-Desc: Cband-10m",
            None: "No Layer Excluded",
        }

        self.sattelite_map = {
            "S1": [i for i in range(11, 15)],
            "S2": [i for i in range(0, 11)],
        }

        if start_time == None:
            self.start_time = dt.now().strftime("%Y_%m_%d:%H_%M_%S")

        else:
            self.start_time = start_time

        try:
            os.mkdir("model_outputs/{}".format(self.start_time))
        except:
            pass

        self.use_gpu = use_gpu
        self.exclude_layer = exclude_layer
        self.num_dpoints = num_dpoints
        self.num_input_channels = num_input_channels
        self.fold_loaders = None

        if torch.cuda.is_available() and self.use_gpu == True:
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        print(self)

    def load_single_SentinelDataset(
        self,
        fpath="subset.csv",
        max_chips=None,
        dir_tiles="data/sentinel/",
        dir_target="data/train_agbm/",
    ):
        dataset = dl.SentinelDataset(
            tile_file=fpath,
            dir_tiles=dir_tiles,
            dir_target=dir_target,
            max_chips=max_chips,
            transform=None,
            device=self.device,
        )

        return dataset

    def load_all_data_w_kfold(
        self, paths, exclude_layer=None, num_dpoints=100, num_folds=5
    ):
        # Convert the data to PyTorch tensors

        input_ = []
        target_ = []

        for path in paths:
            dataset = self.load_single_SentinelDataset(**path)

            if self.num_dpoints == None:
                self.num_dpoints = len(dataset)

            for i in tqdm(range(self.num_dpoints)):
                all_channels = [channel for channel in dataset.__getitem__(i)["image"]]

                if self.exclude_layer != None:
                    if isinstance(self.exclude_layer, list):
                        for idx, c in enumerate(self.exclude_layer):
                            all_channels.pop(c - idx)
                    else:
                        all_channels.pop(self.exclude_layer)

                input_.append(torch.stack(all_channels))
                target_.append(dataset.__getitem__(i)["label"])

        if self.fold_loaders == None:
            kf = KFold(num_folds)
            fold_loaders = {f: {} for f in range(num_folds)}

            for i, (train_index, test_index) in enumerate(kf.split(input_)):
                fold_loaders[i]["train"] = train_index
                fold_loaders[i]["val"] = test_index

        else:
            fold_loaders = self.fold_loaders

        return fold_loaders, input_, target_

    def create_dataloader_from_indexes(self, fold_dict, input_, target_):
        train_index = fold_dict["train"]
        test_index = fold_dict["val"]

        input_tensor = torch.stack([input_[idx] for idx in train_index])
        target_tensor = torch.stack([target_[idx] for idx in train_index])

        # Create a PyTorch data loader
        train_data = torch.utils.data.TensorDataset(input_tensor, target_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=True
        )

        input_tensor = torch.stack([input_[idx] for idx in test_index])
        target_tensor = torch.stack([target_[idx] for idx in test_index])

        # Create a PyTorch data loader
        val_data = torch.utils.data.TensorDataset(input_tensor, target_tensor)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

        return train_loader, val_loader

    # Define the training function
    def train(self, param, folds, input_, target_, num_epochs):
        self.num_epochs = num_epochs

        fold_metrics = {
            "fold": [],
            "epoch": [],
            "train_loss_median": [],
            "val_loss_median": [],
            "train_loss_mean": [],
            "val_loss_mean": [],
            "train_loss_std": [],
            "val_loss_std": [],
        }

        for fold, fold_dict in folds.items():
            model = CNN(**param)
            model.to(self.device)

            # Define the loss function and optimizer
            criterion = nn.MSELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=0.02)

            print("Creating DataLoaders for Fold #{}".format(fold))
            train_loader, val_loader = self.create_dataloader_from_indexes(
                fold_dict, input_, target_
            )

            validation_metrics = []
            train_metrics = []
            for epoch in tqdm(range(num_epochs)):
                running_loss = 0.0
                cur_losses = []
                val_losses = []

                for i, (inputs, labels) in tqdm(enumerate(train_loader)):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    train_loss = np.round(np.sqrt(loss.item()), 5)
                    cur_losses.append(train_loss)

                for i, (val_X, val_y) in tqdm(enumerate(val_loader)):
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device)

                    outputs_val = model(val_X)
                    loss_val = criterion(outputs_val, val_y)
                    val_loss = np.round(np.sqrt(loss_val.item()), 5)
                    val_losses.append(val_loss)

                fold_metrics["fold"].append(fold)
                fold_metrics["epoch"].append(epoch)

                fold_metrics["train_loss_median"].append(np.median(cur_losses))
                fold_metrics["val_loss_median"].append(np.median(val_losses))

                fold_metrics["train_loss_mean"].append(np.mean(cur_losses))
                fold_metrics["val_loss_mean"].append(np.mean(val_losses))

                fold_metrics["train_loss_std"].append(np.std(cur_losses))
                fold_metrics["val_loss_std"].append(np.std(val_losses))

                cur_loss = cur_losses
                train_metrics += cur_loss
                validation_metrics += val_losses

                print(
                    "Epoch [%d/%d], Loss: %.4f"
                    % (epoch + 1, num_epochs, np.mean(cur_loss))
                )

        return fold_metrics, model

    def setup_training(self, folds, input_, target_, num_input_channels=14) -> None:
        # Define the model parameters for each trial

        self.params = [
            {
                "num_input_channels": num_input_channels,
                "conv_filters1": 4,
                "conv_filters2": 8,
                "conv_filters3": 4,
            },
            {
                "num_input_channels": num_input_channels,
                "conv_filters1": 8,
                "conv_filters2": 16,
                "conv_filters3": 16,
            },
            {
                "num_input_channels": num_input_channels,
                "conv_filters1": 16,
                "conv_filters2": 32,
                "conv_filters3": 16,
            },
        ]

        # Create a CNN instance

        self.results = dict()
        self.models = dict()

        for idx, param in enumerate(self.params):
            # Train the model
            train_metrics, model = self.train(
                param, folds, input_, target_, num_epochs=10
            )

            self.results[idx] = pd.DataFrame(train_metrics)
            self.models[idx] = model

        return

    def load_all_datasets(self, paths):
        datasets = defaultdict(dict)

        if self.exclude_layer in self.sattelite_map.keys():
            channel_name = self.exclude_layer
            self.exclude_layer = self.sattelite_map[channel_name]

        elif self.exclude_layer != None:
            channel_name = self.channel_map[self.exclude_layer]
        else:
            channel_name = "None"

        print("CREATING DATASET WITHOUT CHANNEL", self.exclude_layer, channel_name)

        fold_loaders, input_, target_ = self.load_all_data_w_kfold(
            paths=paths,
            exclude_layer=self.exclude_layer,
            num_dpoints=self.num_dpoints,
            num_folds=5,
        )

        datasets[channel_name] = fold_loaders

        return fold_loaders, input_, target_

    def save_results(self) -> None:
        timestamp = dt.now().strftime("%Y_%m_%d:%H_%M_%S")
        # TODO fix lige det her
        os.mkdir("model_outputs/{}/{}".format(self.start_time, timestamp))

        for p in range(len(self.params)):
            if isinstance(self.exclude_layer, list):
                el_name = ", ".join(
                    self.channel_map[channel] for channel in self.exclude_layer
                )
            else:
                el_name = self.channel_map[self.exclude_layer]

            cur_path = "model_outputs/{}/{}/{}".format(self.start_time, timestamp, p)
            os.mkdir(cur_path)

            params_to_save = {**self.params[p]}
            params_to_save["num_dpoints"] = self.num_dpoints
            params_to_save["exclude_layer"] = self.exclude_layer
            params_to_save["exclude_layer_name"] = el_name
            params_to_save["num_epochs"] = self.num_epochs

            with open(cur_path + "/parameters.json", "w") as fp:
                json.dump(params_to_save, fp)

            self.results[p].to_csv(cur_path + "/results.csv", index=None)
            torch.save(self.models[p].state_dict(), cur_path + "/model_state_dict")

        return

    def run(self, fpath=None):
        if fpath is None:
            paths = [
                {
                    "fpath": "large_subset.csv",
                    "max_chips": None,
                    "dir_tiles": "large_sample/sentinel/",
                    "dir_target": "large_sample/target/",
                }
            ]
        else:
            paths = [
                {
                    "fpath": fpath,
                    "max_chips": None,
                    "dir_tiles": "large_sample/sentinel/",
                    "dir_target": "large_sample/target/",
                }
            ]

        folds, input_, target_ = self.load_all_datasets(paths)
        self.folds, self.input_, self.target_ = folds, input_, target_

        results = self.setup_training(
            folds, input_, target_, num_input_channels=self.num_input_channels
        )

        return results


def handle_args():
    args = sys.argv

    if len(args) > 1:
        if args[-1] == "--run_all":
            return {
                "exclude_layer": "run_all",
                "num_input_channels": 14,
                "start_time": dt.now().strftime("%Y_%m_%d:%H_%M_%S"),
            }

        if args[-1] == "--satellite" or args[-1] == "-S":
            return {
                "exclude_layer": "satellite",
                "num_input_channels": 14,
                "start_time": dt.now().strftime("%Y_%m_%d:%H_%M_%S"),
            }

        else:
            return {
                "exclude_layer": int(args[1]),
                "num_input_channels": 14,
                "start_time": dt.now().strftime("%Y_%m_%d:%H_%M_%S"),
            }

    return {
        "exclude_layer": None,
        "num_input_channels": 15,
        "start_time": dt.now().strftime("%Y_%m_%d:%H_%M_%S"),
    }


def run_all(args):
    """Exclude one band at a time to perform the ablation study"""

    args["exclude_layer"] = None
    args["num_input_channels"] = 15

    AGBM_trainer = AGBM_CNN(**args)
    print(AGBM_trainer.run())

    AGBM_trainer.save_results()

    folds = AGBM_trainer.folds

    for i in tqdm(range(15)):
        args["exclude_layer"] = i
        args["num_input_channels"] = 14

        AGBM_trainer = AGBM_CNN(**args)
        AGBM_trainer.folds = folds
        AGBM_trainer.run()
        AGBM_trainer.save_results()


def satellite_run(args):
    args["exclude_layer"] = None
    args["num_input_channels"] = 15

    AGBM_trainer = AGBM_CNN(**args)
    AGBM_trainer.run()
    AGBM_trainer.save_results()
    folds = AGBM_trainer.folds

    sat_map = AGBM_trainer.sattelite_map

    for satellite_id, channels in tqdm(sat_map.items()):
        args["exclude_layer"] = satellite_id
        args["num_input_channels"] = 15 - len(channels)

        AGBM_trainer = AGBM_CNN(**args)
        AGBM_trainer.fold_loaders = folds
        AGBM_trainer.run()
        AGBM_trainer.save_results()
        folds = AGBM_trainer.folds


if __name__ == "__main__":
    args = handle_args()

    args["num_dpoints"] = None

    if args["exclude_layer"] == "run_all":
        run_all(args)

    elif args["exclude_layer"] == "satellite":
        satellite_run(args)

    else:
        month_files = [
            "0.csv",  # september
            "1.csv",  # october
            "2.csv",  # november
            "3.csv",  # december
            "4.csv",  # january
            "5.csv",  # february
            "6.csv",  # march
            "7.csv",  # april
            "8.csv",  # may
            "9.csv",  # june
            "10.csv",  # july
            "11.csv",  # august
        ]

        for month in month_files:
            AGBM_trainer = AGBM_CNN(**args)
            AGBM_trainer.run(fpath=month)
            AGBM_trainer.save_results()
