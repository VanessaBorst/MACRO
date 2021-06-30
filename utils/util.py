import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(path_to_file):
    file_path = Path(path_to_file)
    with file_path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, path_to_file):
    file_path = Path(path_to_file)
    with file_path.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def plot_record_from_df(record_name, df_record, preprocesed=False):
    print(record_name)
    fig, axs = plt.subplots(6, 2, figsize=(15, 15), constrained_layout=True)
    title = "Record " + record_name + " after padding" if preprocesed else "Record " + record_name + " before padding"
    fig.suptitle(title)
    axis_0 = 0
    axis_1 = 0
    for lead in df_record.columns:
        lead_data = df_record[lead].to_list()
        axs[axis_0, axis_1].plot(lead_data)
        axs[axis_0, axis_1].set_title(lead)
        axis_0 = (axis_0 + 1) % 6
        if axis_0 == 0:
            axis_1 += 1
    plt.show()


def plot_record_from_np_array(record_data, num_rows=6, num_cols=2):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15), constrained_layout=True)
    axis_0 = 0
    axis_1 = 0
    for lead_idx in range(0, len(record_data)):
        lead_data = record_data[lead_idx]
        axs[axis_0, axis_1].plot(lead_data)
        axs[axis_0, axis_1].set_title("Lead-ID: " + str(lead_idx))
        axis_0 = (axis_0 + 1) % 6
        if axis_0 == 0:
            axis_1 += 1
    plt.show()


def plot_grad_flow_lines(named_parameters, ax):
    with torch.no_grad():
        ave_grads = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                ave_grads.append(p.grad.detach().abs().mean().cpu().numpy())
        ax.plot(ave_grads, alpha=0.3, color="b")


def plot_grad_flow_bars(named_parameters, ax):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters(), fig_gradient_flows)" to visualize the gradient flow
    At the end of the epoch, send the Figure to the TensorboardWriter'''

    with torch.no_grad():
        ave_grads = []
        max_grads = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                ave_grads.append(p.grad.detach().abs().mean().cpu().numpy())
                max_grads.append(p.grad.detach().abs().max().cpu().numpy())

        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
