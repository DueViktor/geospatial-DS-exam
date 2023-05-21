import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import torch
import torch.nn as nn
from tqdm import tqdm

import data_loader as dl

months = [
    "Sept",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "March",
    "April",
    "May",
    "June",
    "July",
    "Aug",
]

# Simple plotting utils made complicated


def aggregate_result(df, index):
    agg_df = df.groupby("epoch").mean()
    agg_df.drop("fold", axis=1, inplace=True)
    agg_df["config_index"] = index
    return agg_df


def plot_single_config(data_dir, idx=0):
    config_dir = os.listdir(data_dir)[idx]

    dfs = {}
    configurations = {}
    for config_index in range(len(os.listdir(data_dir + config_dir))):
        df = pd.read_csv(
            data_dir + config_dir + "/" + str(config_index) + "/results.csv"
        )
        dfs[config_index] = df

        with open(
            data_dir + config_dir + "/" + str(config_index) + "/parameters.json", "r"
        ) as file:
            for line in file:
                conf = json.loads(line)

        configurations[config_index] = conf

    for cidx, conf in configurations.items():
        print(cidx)
        print(conf)
        print()
        print()
        print()

    plt.style.use("fivethirtyeight")
    for index, df in dfs.items():
        agg_df = aggregate_result(df, index)

        agg_df[["train_loss_median", "val_loss_median"]].plot(figsize=[16, 9])
        plt.ylim([60, 100])
        plt.show()


def get_best_model(idx, data_dir, eval_metrix="val_loss_median", w_index=False):
    # idx references the index of the global configuration (i.e. ablation-setting)
    # model_outputs/study/[ablation_idx]/[config_idx]
    # This means that this function returns the best model for a given ablation study (fx excluding channel 0 or channel 1 or satellite 2...)

    config_dir = [i for i in sorted(os.listdir(data_dir)) if '2023' in i][idx]
    params_dirs = [i for i in sorted(os.listdir(data_dir+'/'+config_dir)) if '.DS_Store' not in i]
    configs = len(params_dirs)
    dfs = {}
    configurations = {}

    
    for config_index in range(configs):
        
        df = pd.read_csv(
            data_dir + config_dir + "/" + str(config_index) + "/results.csv"
        )
        dfs[config_index] = df

        with open(
            data_dir + config_dir + "/" + str(config_index) + "/parameters.json", "r"
        ) as file:
            for line in file:
                conf = json.loads(line)

        configurations[config_index] = conf

    cur_min = 1000
    cur_min_index = 99
    best_agg = None
    for index, df in dfs.items():
        agg_df = aggregate_result(df, index)
        m = agg_df[eval_metrix].min()
        if m < cur_min:
            cur_min = m
            cur_min_index = index
            best_agg = agg_df

    if w_index:
        return best_agg, configurations[cur_min_index], cur_min_index
    else:
        return best_agg, configurations[cur_min_index]


def plot_best_config(data_dir, idx=0):
    best_agg, conf = get_best_model(idx, data_dir)

    print(conf)

    plt.style.use("fivethirtyeight")

    best_agg[["train_loss_median", "val_loss_median"]].plot(figsize=[16, 9],title='test/validation RMSE over 10 epochs')
    plt.ylim([70, 82])
    plt.savefig('assets/best_config.png')
    plt.show()


def performance_plots(data_dir, max_num=-1):
    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=[16, 9])
    range_ = len(os.listdir(data_dir)[:max_num])
    for idx in range(range_):
        res, conf = get_best_model(idx, data_dir)
        res["val_loss_median"].plot(linewidth=2, color="green")
        plt.ylim([60, 100])
    plt.title("Performance of different models in " + data_dir)
    plt.show()


def plot_ablation(
    data_dir,
    metric="val_loss_median",
    id_="exclude_layer_name",
    title="sometitle",
    vline=None,
):
    result_dict = {}
    param_dict = {}

    average_validation_rmse_final_epoch = []
    excluded_channel = []

    configs = [i for i in os.listdir(data_dir) if '2023' in i]

    num_configs = len(configs)

    for config_index in range(num_configs):
        results, params = get_best_model(config_index, data_dir)

        result_dict[config_index] = results
        param_dict[config_index] = params

        final_epoch = params["num_epochs"] - 1
        perf = results.iloc[final_epoch]["val_loss_median"]

        average_validation_rmse_final_epoch.append(perf)
        excluded_channel.append(params[id_])

    plt.style.use("fivethirtyeight")

    if id_ == 'exclude_layer':
        mapping = {"None":"None",
                   str([11, 12, 13, 14]):'S1',
                   str([i for i in range(11)]):'S2'}
        excluded_channel_ = [mapping[str(i)] for i in excluded_channel]
        excluded_channel = excluded_channel_
    plot_df = pd.DataFrame(
        {
            "excluded_channel": excluded_channel,
            "r-mse": average_validation_rmse_final_epoch,
        }
    ).sort_values("r-mse", ascending=False)


    plot_df.plot(
        kind="barh", x="excluded_channel", figsize=[16, 9], xlim=[40, 95], color="green"
    )

    if vline:
        plt.vlines(
            plot_df.loc[plot_df["excluded_channel"] == vline]["r-mse"].values[0],
            -1,
            99,
            color="red",
            linewidth=1.5,
        )
    plt.title(title)
    plt.tight_layout()
    plt.savefig("assets/{}.png".format(title))

    plt.show()

    return param_dict, result_dict


# Code to fetch dataset in correct format


def load_single_SentinelDataset(
    fpath="large_subset.csv",
    max_chips=None,
    dir_tiles="large_sample/sentinel/",
    dir_target="large_sample/target/",
    dir_test="large_sample/test_features/",
):
    dataset = dl.SentinelDataset(
        tile_file=fpath,
        dir_tiles=dir_tiles,
        dir_target=dir_target,
        max_chips=max_chips,
        transform=None,
        device="cpu",
    )

    return dataset


def create_dataloader_from_indexes(indexes, fpath, exclude_layer=None, num_dpoints=100):
    input_ = []
    target_ = []

    dataset = load_single_SentinelDataset(fpath=fpath, max_chips=num_dpoints)
    if num_dpoints == None:
        num_dpoints = len(dataset)
    for i in tqdm(range(num_dpoints)):
        all_channels = [channel for channel in dataset.__getitem__(i)["image"]]
        if exclude_layer != None:
            if isinstance(exclude_layer, list):
                for idx, c in enumerate(exclude_layer):
                    all_channels.pop(c - idx)
            else:
                all_channels.pop(exclude_layer)

        input_.append(torch.stack(all_channels))
        target_.append(dataset.__getitem__(i)["label"])

    input_tensor = torch.stack([input_[idx] for idx in indexes])
    target_tensor = torch.stack([target_[idx] for idx in indexes])

    # Create a PyTorch data loader
    test_data = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    return test_loader


def predict(model,indexes,num_dpoints=100, fpath="large_subset.csv"):
    metrics = dict()
    # Define the loss function and optimizer
    criterion = nn.MSELoss(reduction="mean")
    loader = create_dataloader_from_indexes(indexes, fpath, num_dpoints=num_dpoints)

    cur_losses = []
    for i, (inputs, labels) in tqdm(enumerate(loader)):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        train_loss = np.round(np.sqrt(loss.item()), 5)
        cur_losses.append(train_loss)

    metrics["train_loss_median"] = np.median(cur_losses)
    metrics["train_loss_mean"] = np.mean(cur_losses)
    metrics["train_loss_std"] = np.std(cur_losses)

    return metrics


def single_predict(model, chip_id, df, fpath):
    criterion = nn.MSELoss(reduction="mean")

    ix = df.loc[df["chipid"] == chip_id].index.values[0]

    loader = create_dataloader_from_indexes([ix], fpath, num_dpoints=100)

    cur_losses = []
    for i, (inputs, labels) in tqdm(enumerate(loader)):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        train_loss = np.round(np.sqrt(loss.item()), 5)

        return outputs.detach().numpy(), train_loss


# Code to select the best performing model from a certain study

import torch

from train_CNN import CNN


def get_model_args(args):
    CNN_params = [
        "num_input_channels",
        "conv_filters1",
        "conv_filters2",
        "conv_filters3",
    ]
    return {p: args[p] for p in CNN_params}


def get_best_model_index(idx, data_dir, eval_metrix="val_loss_median"):
    # idx references the index of the global configuration (i.e. ablation-setting)
    # model_outputs/study/[ablation_idx]/[config_idx]
    # This means that this function returns the best model for a given ablation study (fx excluding channel 0 or channel 1 or satellite 2...)

    config_dir = [i for i in sorted(os.listdir(data_dir)) if '2023' in i][idx]
    params_dirs = [i for i in sorted(os.listdir(data_dir+'/'+config_dir)) if '.DS_Store' not in i]
    configs = len(params_dirs)
    dfs = {}
    configurations = {}
    for config_index in range(configs):
        df = pd.read_csv(
            data_dir + config_dir + "/" + str(config_index) + "/results.csv"
        )
        dfs[config_index] = df

        with open(
            data_dir + config_dir + "/" + str(config_index) + "/parameters.json", "r"
        ) as file:
            for line in file:
                conf = json.loads(line)

        configurations[config_index] = conf

    plt.style.use("fivethirtyeight")

    cur_min = 1000
    cur_min_index = 99
    best_agg = None
    for index, df in dfs.items():
        agg_df = aggregate_result(df, index)
        m = agg_df[eval_metrix].min()
        if m < cur_min:
            cur_min = m
            cur_min_index = index

    return cur_min_index


def get_model_path(ablation_index, data_dir, bmx=None):
    if not bmx:
        bmx = get_best_model_index(ablation_index, data_dir)

    config_dir = [i for i in sorted(os.listdir(data_dir)) if '2023' in i][ablation_index]

    return data_dir + config_dir + "/" + str(bmx) + "/model_state_dict"


# Create random monthly datasets
def create_monthly_datasets(num_chips=100):
    ls = (
        pd.read_csv("large_subset.csv", index_col=0)
        .drop_duplicates()
        .reset_index()
        .drop("index", axis=1)
        .sort_index(ascending=False)
    )

    ls.sort_index(inplace=True)

    indexes = defaultdict(list)
    for chip in range(num_chips):
        random_chip = random.choice(ls["chipid"].unique())

        for month in range(12):
            ix = ls.loc[
                (ls["month"] == month) & (ls["chipid"] == random_chip)
            ].index.values[0]

            indexes[month].append(ix)

        for month in range(12):
            ls.iloc[indexes[month]].to_csv("subsets/subset_{}_{}.csv".format(num_chips, month))


def get_model_from_path(data_dir, ablation_index=0):
    # Chose a random model (e.g. excluding some layer)
    # standard is to use the full model

    results, params, ix = get_best_model(ablation_index, data_dir, w_index=True)
    
    PATH = get_model_path(ix, data_dir)
    model_args = get_model_args(params)
    model = CNN(**model_args)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model


def evaluate_monthly(model):
    perfs = []

    months = [
        "Sept",
        "Oct",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "March",
        "April",
        "May",
        "June",
        "July",
        "Aug",
    ]

    for month in range(12):
        print("EVALUATING MODEL ON", months[month])
        fpath = "subsets/subset_100_{}.csv".format(month)
        perfs.append(
            predict(
                model, indexes=[i for i in range(100)], num_dpoints=100, fpath=fpath
            )
        )

    y = [p["train_loss_median"] for p in perfs]

    plt.figure(figsize=[16, 9])
    plt.bar(x=months, height=y, color="green")
    plt.title("Performance on different months")
    plt.tight_layout()
    plt.savefig("assets/months.png")
    plt.show()


def evaluate_monthly_multiple(models):
    perfs = []

    months = [
        "Sept",
        "Oct",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "March",
        "April",
        "May",
        "June",
        "July",
        "Aug",
    ]

    for month in range(12):
        print("EVALUATING MODEL ON", months[month])
        fpath = "subsets/subset_100_{}.csv".format(month)
        model = models[month]
        perfs.append(
            predict(
                model, indexes=[i for i in range(100)], num_dpoints=100, fpath=fpath
            )
        )

    y = [p["train_loss_median"] for p in perfs]

    plt.figure(figsize=[16, 9])
    plt.bar(x=months, height=y, color="green")
    plt.title("RMSE performance on all months")
    plt.tight_layout()
    plt.savefig("assets/months_mult.png")
    plt.show()


def get_best_model_pr_month(data_dir):
    models = []
    for month in range(0, 12):
        model = get_model_from_path(data_dir, month)
        models.append(model)
    return models


# Load single target datapoint
def _read_tif_to_tensor(tif_path):
    with rasterio.open(tif_path) as src:
        X = torch.tensor(
            src.read().astype(np.float32),
            dtype=torch.float32,
            device="cpu",
            requires_grad=False,
        )
    return X


# HJÆLP
def barplot_3D(t,ax1,title=''):
    num_vals = 100

    mat = np.matrix(t.reshape([256, 256]))

    mat = mat[:num_vals, :num_vals]

    z = []
    for row in np.array(mat).tolist():
        for i in row:
            z.append(i)

    # set up the figure and axes
    #fig = plt.figure(figsize=(16, 9))
    #ax1 = fig.add_subplot(121, projection='3d')

    # fake data
    _x = np.arange(num_vals)
    _y = np.arange(num_vals)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = np.array(z)
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True,color = 'green')
    ax1.set_title('BioMass '+title)
    
    #return ax1




def plot_agbm(t=None,chip_id=None,model=None,df=None,fpath=None,month=''):
    if not t and not chip_id:
        return "INPUT DATA YOU DUMBDUMB"

    elif chip_id:
        t = _read_tif_to_tensor('large_sample/target/{}_agbm.tif'.format(chip_id))
        m = max(t.flatten())
        m +=m/10
    elif t:
        pass

    
    if model!=None:
        t_est,score = single_predict(model,chip_id,df,fpath)
        
        ts = [t,t_est]
        titles = [f'Ground Truth ({chip_id})',f'Estimate ({month}) | RMSE: '+str(round(score,2))]
        
        fig,axes = plt.subplots(1,2,subplot_kw=dict(projection='3d'),figsize=[16,9])
        for i in range(len(axes)):
            barplot_3D(ts[i],axes[i],title=titles[i])
            
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.2)
        plt.savefig(f'assets/{month}.png')
        # Show the plots
        plt.show()

        #barplot_3D(t)
        #barplot_3D(t_est)
        print('ERROR',score)
    else:
        barplot_3D(t)




def get_subset_from_month(month,chip_id=None):
    index = months.index(month)
    fpath = "subsets/subset_100_{}.csv".format(index)
    df = pd.read_csv(fpath).drop("Unnamed: 0", axis=1)
    if chip_id and chip_id in df["chipid"].values.tolist():
        return chip_id, fpath, df
    else:
        chip_id = df["chipid"].values.tolist()[0]
        return chip_id, fpath, df



def plot_input_data(chip_id):
    channel_map = ["S2-B2: Blue-10m",
                "S2-B3: Green-10m",
                "S2-B4: Red-10m",
                "S2-B5: VegRed-704nm-20m",
                "S2-B6: VegRed-740nm-20m",
                "S2-B7: VegRed-780nm-20m",
                "S2-B8: NIR-833nm-10m",
                "S2-B8A: NarrowNIR-864nm-20m",
                "S2-B11: SWIR-1610nm-20m",
                "S2-B12: SWIR-2200nm-20m",
                "S2-CLP: Clouse_gpuudProb-160m",
                "S1-VV-Asc: Cband-10m",
                "S1-VH-Asc: Cband-10m",
                "S1-VV-Desc: Cband-10m",
                "S1-VH-Desc: Cband-10m"]

    

    fig,axes = plt.subplots(4,4,figsize=[29,20])
    
    ch = 0
    cv = 0
    cc = 0

    fontdict = {'fontsize': 9,
                'fontweight': 'light'}

    fp = f'large_sample/sentinel/{chip_id}_S2_00.tif'
    img = rasterio.open(fp)
    for i in range(1,12):
        name = channel_map[cc]
        ax = axes[ch][cv]
        show(img.read(i),title=name,ax=ax)
        if ch < 3:
            ch+=1
            cc+=1
        else:
            ch=0
            cv+=1
            cc+=1


    fp = f'large_sample/sentinel/{chip_id}_S1_00.tif'
    img = rasterio.open(fp)
    for i in range(1,5):
        name = channel_map[cc]
        ax = axes[ch][cv]
        show(img.read(i),title=name,ax=ax)
        if ch < 3:
            ch+=1
            cc+=1
        else:
            ch=0
            cv+=1
            cc+=1
    

    fp = f'large_sample/target/{chip_id}_agbm.tif'
    img = rasterio.open(fp)
    name = 'Target Values'
    ax = axes[ch][cv]
    show(img.read(),title=name,ax=ax)

    #fig.delaxes(axes[ch,cv]) 
    plt.suptitle(f"Input and target data for chip {chip_id}",fontsize=34)
    
    plt.savefig(f'assets/input_{chip_id}.png')