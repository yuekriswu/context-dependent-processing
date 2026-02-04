import copy
import logging
import argparse

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from niarb.cli import fit
from niarb.dataset import Dataset
from niarb.tensors import categorical
from niarb import neurons, nn, perturbation, random
from niarb.cell_type import CellType
from niarb.nn.modules import frame
import scipy.io as sio
from niarb.optimize import constraint

config = 'classical_only'

def calc_distance_matrix_pbc(grid_width=30, grid_height=30):
    N = grid_width * grid_height

    # Create 2D indices for all neurons
    indices = torch.arange(N)
    x = indices // grid_height  # row index
    y = indices % grid_height  # column index

    # Stack to get (N, 2) array of coordinates
    coords = torch.stack([x, y], dim=1)  # shape: (N, 2)

    # Compute pairwise differences
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)

    # Apply periodic boundary conditions
    diff[..., 0] = torch.minimum(diff[..., 0].abs(), grid_width - diff[..., 0].abs())
    diff[..., 1] = torch.minimum(diff[..., 1].abs(), grid_height - diff[..., 1].abs())

    # Compute Euclidean distance
    dist_matrix = diff.pow(2).sum(dim=-1).sqrt()  # shape: (N, N)

    return dist_matrix

def read_rf_parameters_X(stim_type_idx):
    para_X = torch.zeros(4, 9)
    if stim_type_idx == 0:
        para_X[0, :] = 1/8 * torch.sqrt(torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85]))
        para_X[2, :] = torch.full((9,), 15.0)
        para_X[3, :] = torch.full((9,), 15.0)
    else:
        para_X[0, :] = 1/2 * torch.sqrt(torch.tensor([45, 45, 45, 45, 45, 35, 25, 15, 5]))
        para_X[2, :] = torch.full((9,), 20.0)
        para_X[3, :] = torch.full((9,), 20.0)
    return para_X

def define_L4_inputs(stim_type, stim_idx):
    R_L4_np = sio.loadmat('../data/rate_field_inputs_L4.mat')['rate_field_inputs']
    R_L4_total = torch.from_numpy(R_L4_np).float()
    R_L4 = R_L4_total[stim_type * 9 + stim_idx, :, :] - torch.tensor(0.6092)

    H_L4 = torch.sqrt(torch.tensor(0.6092) + R_L4) - torch.sqrt(torch.tensor(0.6092))
    return H_L4

def define_LM_inputs(stim_type, stim_idx):
    R_LM_np = sio.loadmat('../data/rate_field_inputs_LM.mat')['rate_field_inputs']
    R_LM_total = torch.from_numpy(R_LM_np).float()
    R_LM = R_LM_total[stim_type * 9 + stim_idx, :, :] - torch.tensor(0.5996)

    H_LM = torch.sqrt(torch.tensor(0.5996) + R_LM) - torch.sqrt(torch.tensor(0.5996))
    return H_LM

def define_X_Y_inputs(para_X, stim_idx, dist_mat):
    # rate field related parameters
    sigma2X2_1, sigma2X2_2 = 2.0 * para_X[2, stim_idx] ** 2, 2.0 * para_X[3, stim_idx] ** 2
    R_X = para_X[0, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2X2_1) - para_X[1, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2X2_2)
    H_X = torch.sqrt(torch.tensor(0.5996) + R_X) - torch.sqrt(torch.tensor(0.5996))
    H_Y = torch.zeros_like(H_X)
    return H_X, H_Y

def setup():
    cell_types = [
        CellType.PYR,
        CellType.PV,
        CellType.SST,
        CellType.VIP,
        CellType.L4,
        CellType.LM,
        CellType.X,
        CellType.Y,
    ]

    N_space = (
        30,
        30,
    )  # number of neurons along each spatial dimension, the origin is (15, 15)
    space_extent = (
        180,
        180,
    )  # in degrees, grid spacing 6 degrees, therefore space is -90 to 90.

    # setup a network with multiple interneuron cell types

    x = neurons.as_grid(
        n=len(cell_types),
        N_space=N_space,
        cell_types=cell_types,
        space_extent=space_extent,
    )

    # calculate distance of each neuron to origin in degree
    x["distance"] = x["space"].norm(
        dim=-1
    )  # (4, *N_space), the first point is (-90, -90), and last one is (90, 90)

    diff_Gaussian_mask = torch.ones(8, 8, 30, 30)
    diff_Gaussian_mask[0, 5, :, :] = 1

    diff_Gaussian_mask2 = torch.zeros(8, 8, 30, 30)
    diff_Gaussian_mask2[0, 5, :, :] = 0

    # define V1 model
    model = nn.V1(
        ["cell_type", "space"],  # connectivity depends on both cell type and space
        cell_types=cell_types,
        f=nn.SSN(),  # SSN nonlinearity
        tau=[
            1.0,
            0.5,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
        ],  # relative time constants of E, PV, SST, and VIP
        mode="numerical",  # run numerical simulation
        space_strength_kernel=nn.Gaussian,  # Gaussian spatial connectivity kernel
        space_strength_kernel2=nn.Gaussian,  # Gaussian spatial connectivity kernel
        diff_Gaussian_mask=diff_Gaussian_mask,
        diff_Gaussian_mask2=diff_Gaussian_mask2,
        vf_symmetry=False,  # allowing to specify different vf for different cell types, vf: baseline voltage/current for different cell types
        vf_optim=False,
        sigma_optim=False,
    )

    # set model parameters
    state_dict = {
        "sigma": torch.tensor(
            [
                [7, 5, 7, 5, 7, 15, 15, 15],
                [5, 4, 5, 4, 7, 15, 15, 15],
                [7, 5, 7, 5, 7, 15, 15, 15],
                [5, 4, 7, 4, 7, 15, 15, 15],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),

        "sigma2": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1], # the only value that matters is one that is not 1.
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),

        "vf": torch.tensor(
            [
                np.sqrt(0.5996),
                np.sqrt(1.35),
                np.sqrt(0.6818),
                np.sqrt(0.8437),
                np.sqrt(0.6092),
                np.sqrt(0.5996),
                np.sqrt(0.5996),
                np.sqrt(0.5996),
            ]
        ),
    }
    model.load_state_dict(state_dict, strict=False)  # what is the strict?
    return x, model  # x: input, and model: recurrent model

def set_tags():
    # set the stimulus condition tags, c: classical, i: inverse, number: stimulus size
    if config == 'classical_only':
        tags = frame.ParameterFrame(
            {
                "size": torch.tensor(list(range(5, 86, 10))),
                "inverse": torch.tensor([False] * 9),
            }
        )
    elif config == 'inverse_only':
        tags = frame.ParameterFrame(
            {
                "size": torch.tensor(list(range(5, 86, 10))),
                "inverse": torch.tensor([True] * 9),
            }
        )
    else:
        tags = frame.ParameterFrame(
            {
                "size": torch.tensor(list(range(5, 86, 10)) * 2),
                "inverse": torch.tensor([False] * 9 + [True] * 9),
            }
        )
    return tags

def get_experimental_data(x):
    data = []

    dist_mat = x["distance"][0]
    intervals = pd.interval_range(start=0, end=132, freq=6, closed="left")
    distances_flat = dist_mat.flatten().cpu().numpy()
    unique_distances = np.unique(distances_flat)

    # Get the interval index for each distance
    interval_indices = intervals.get_indexer(unique_distances)
    unique_distances = torch.tensor(unique_distances)
    interval_sorted = intervals[interval_indices]

    activity_E = torch.tensor(sio.loadmat('../data/rate_field_activity_E.mat')['rate_field_activity']) - torch.tensor(0.5996)
    activity_PV = torch.tensor(sio.loadmat('../data/rate_field_activity_PV.mat')['rate_field_activity']) - torch.tensor(1.35)
    activity_SST = torch.tensor(sio.loadmat('../data/rate_field_activity_SST.mat')['rate_field_activity']) - torch.tensor(0.6818)
    activity_VIP = torch.tensor(sio.loadmat('../data/rate_field_activity_VIP.mat')['rate_field_activity']) - torch.tensor(0.8437)

    if config == 'classical_only':
        l_stim_type = [0]
    elif config == 'inverse_only':
        l_stim_type = [1]
    else:
        l_stim_type = [0, 1]

    for i in l_stim_type:

        if i == 0:
            b_inverse = False
        else:
            b_inverse = True

        dr_se = 1
        dr_se_VIP = 0.1

        for j in np.arange(9):

            for k in range(activity_E.shape[1]):
                data.append(
                    pd.DataFrame(
                        {
                            "size": [int(5 + j * 10)],
                            "inverse": [b_inverse],
                            "distance": [interval_sorted[k]],
                            "cell_type": ["PYR"],
                            "dr": [activity_E[i*9+j, k].item()],
                            "dr_se": dr_se,
                        }
                    )
                )
                data.append(
                    pd.DataFrame(
                        {
                            "size": [int(5 + j * 10)],
                            "inverse": [b_inverse],
                            "distance": [interval_sorted[k]],
                            "cell_type": ["PV"],
                            "dr": [activity_PV[i*9+j, k].item()],
                            "dr_se": dr_se,
                        }
                    )
                )
                data.append(
                    pd.DataFrame(
                        {
                            "size": [int(5 + j * 10)],
                            "inverse": [b_inverse],
                            "distance": [interval_sorted[k]],
                            "cell_type": ["SST"],
                            "dr": [activity_SST[i*9+j, k].item()],
                            "dr_se": dr_se,
                        }
                    )
                )
                data.append(
                    pd.DataFrame(
                        {
                            "size": [int(5 + j * 10)],
                            "inverse": [b_inverse],
                            "distance": [interval_sorted[k]],
                            "cell_type": ["VIP"],
                            "dr": [activity_VIP[i*9+j, k].item()],
                            "dr_se": dr_se_VIP,
                        }
                    )
                )

    data = pd.concat(data, ignore_index=True)
    data["distance"] = data["distance"].astype("category")
    data["cell_type"] = data["cell_type"].astype("category")
    return data

def get_inputs(x):
    dist_mat = x["distance"][0]
    if config == 'classical_only':
        inputs = torch.zeros(9, 8, 30, 30)
        l_stim_type = [0]
    elif config == 'inverse_only':
        inputs = torch.zeros(9, 8, 30, 30)
        l_stim_type = [1]
    else:
        inputs = torch.zeros(18, 8, 30, 30)
        l_stim_type = [0, 1]

    for stim_type_idx in l_stim_type:
        para_X = read_rf_parameters_X(stim_type_idx)

        for stim_idx in np.arange(9):
            H_L4 = define_L4_inputs(stim_type_idx, stim_idx)
            H_LM = define_LM_inputs(stim_type_idx, stim_idx)
            H_X, H_Y = define_X_Y_inputs(para_X, stim_idx, dist_mat)

            H_X_reshaped = H_X.reshape(30, 30)
            H_Y_reshaped = H_Y.reshape(30, 30)

            if config == 'classical_only' or 'inverse_only':
                inputs[stim_idx, -4, :, :] = H_L4  # inputs from L4
                inputs[stim_idx, -3, :, :] = H_LM  # inputs from LM
                inputs[stim_idx, -2, :, :] = H_X_reshaped  # inputs from X
                inputs[stim_idx, -1, :, :] = H_Y_reshaped  # inputs from Y
            else:
                inputs[stim_type_idx * 9 + stim_idx, -4, :, :] = H_L4  # inputs from L4
                inputs[stim_type_idx * 9 + stim_idx, -3, :, :] = H_LM  # inputs from LM
                inputs[stim_type_idx * 9 + stim_idx, -2, :, :] = H_X_reshaped  # inputs from X
                inputs[stim_type_idx * 9 + stim_idx, -1, :, :] = H_Y_reshaped  # inputs from Y

    return inputs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--log-level", "--ll", dest="log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    if args.log_level == "DEBUG":
        logging.getLogger("matplotlib").setLevel("INFO")

    x, model = setup()

    tags = set_tags()
    inputs = get_inputs(x)

    data = get_experimental_data(x)
    data = data.groupby(
        ["size", "inverse", "distance", "cell_type", "dr_se"], as_index=False, observed=True
    )["dr"].mean()

    if isinstance(data, pd.DataFrame):
        data = [data]

    dataset = Dataset(neurons=x, inputs=inputs, tags=tags, data=data)

    pipeline = nn.Pipeline(model=model, data=data)
    init_state_dict = nn.state_dict(pipeline)

    init_state_dict = {
        k: v.detach().clone()  # create copy, don't pass by reference
        for k, v in init_state_dict.items()
        if k in {"sigma", "sigma2", "vf"}  # only fix sigma and vf
    }

    constraints = [constraint.EqConstraint1(), constraint.EqConstraint2(), constraint.EqConstraint3(), constraint.EqConstraint4()]

    losses, state_dicts = fit.run(
        data,
        dataset,
        pipeline,
        N=10,
        progress=True,
        out="../optimized_files/optimized_files_gaussian_classical_no_recurrent_excitation",
        loss_threshold=0.44,
        normalized_loss=False,
        init_state_dict=init_state_dict,
        constraints=constraints,
        weighted_loss=True,
    )

if __name__ == "__main__":
    main()
