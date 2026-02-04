import os

import torch
import numpy as np
import pandas as pd

from niarb import neurons, nn
from niarb.cell_type import CellType
from niarb.nn.modules import frame
import scipy.io as sio

config = 'classical_only'
l_fitting_config = ['gaussian_classical_no_X_1', 'gaussian_classical_no_X_2', 'gaussian_classical_no_X_3']

def calc_distance_matrix_pbc(grid_width=30, grid_height=30): # check this one, which one is the center, to do, it matches my simulations, but probably it doesn't match HoYin's
    N = grid_width * grid_height

    # Create 2D indices for all neurons
    indices = torch.arange(N)
    x = indices // grid_height  # row index
    y = indices % grid_height   # column index

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

def setup(model_name, fitting_config):
    file_path = ("/Users/kriswu/Projects/spatialrnn_axon/examples/spatialrnn/optimized_files/optimized_files_") + fitting_config + "/" + model_name
    opti_networks_tensor = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)['gW']
    opti_networks = opti_networks_tensor.detach().cpu().numpy()

    # convert to actual connection strength
    opti_networks[0, :] = opti_networks[0, :] / (2 * np.sqrt(0.5996))
    opti_networks[1, :] = opti_networks[1, :] / (2 * np.sqrt(1.35))
    opti_networks[2, :] = opti_networks[2, :] / (2 * np.sqrt(0.6818))
    opti_networks[3, :] = opti_networks[3, :] / (2 * np.sqrt(0.8437))

    cell_types = [CellType.PYR, CellType.PV, CellType.SST, CellType.VIP, CellType.L4, CellType.LM]

    N_space = (30, 30,)  # number of neurons along each spatial dimension, the origin is (15, 15)
    space_extent = (180, 180,)  # in degrees, grid spacing 6 degrees, therefore space is -90 to 90.

    # setup a network with multiple interneuron cell types
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=N_space,
        cell_types=cell_types,
        space_extent=space_extent,
    )

    # calculate distance of each neuron to origin in degree
    x["distance"] = x["space"].norm(dim=-1)  # (4, *N_space), the first point is (-90, -90), and last one is (90, 90)

    # define V1 model
    model = nn.V1(
        ["cell_type", "space"],  # connectivity depends on both cell type and space
        cell_types=cell_types,
        f=nn.SSN(),  # SSN nonlinearity
        tau=[1.0, 0.5, 0.5, 0.5, 1.0, 1.0],  # relative time constants of E, PV, SST, and VIP
        mode="numerical",  # run numerical simulation
        space_strength_kernel=nn.Gaussian,  # Gaussian spatial connectivity kernel
        vf_symmetry=False,  # allowing to specify different vf for different cell types, vf: baseline voltage/current for different cell types
    )
    state_dict = {
        "gW": torch.tensor(
            [
                [2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 0]),
                 2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 1]),
                 2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 2]),
                 2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 3]),
                 2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 4]),
                 2 * np.sqrt(0.5996) * np.abs(opti_networks[0, 5])],
                [2 * np.sqrt(1.35) * np.abs(opti_networks[1, 0]),
                 2 * np.sqrt(1.35) * np.abs(opti_networks[1, 1]),
                 2 * np.sqrt(1.35) * np.abs(opti_networks[1, 2]),
                 2 * np.sqrt(1.35) * np.abs(opti_networks[1, 3]),
                 2 * np.sqrt(1.35) * np.abs(opti_networks[1, 4]),
                 2 * np.sqrt(1.35) * np.abs(opti_networks[1, 5])],
                [2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 0]),
                 2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 1]),
                 2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 2]),
                 2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 3]),
                 2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 4]),
                 2 * np.sqrt(0.6818) * np.abs(opti_networks[2, 5])],
                [2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 0]),
                 2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 1]),
                 2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 2]),
                 2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 3]),
                 2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 4]),
                 2 * np.sqrt(0.8437) * np.abs(opti_networks[3, 5])],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            ]
        ),

        "sigma": torch.tensor(
            [
                [7, 5, 7, 5, 7, 15],
                [5, 4, 5, 4, 7, 15],
                [7, 5, 7, 5, 7, 15],
                [5, 4, 7, 4, 7, 15],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
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
            ]
        ),
    }

    model.load_state_dict(state_dict, strict=False)
    return x, model  # x: input, and model: recurrent model

def get_inputs():

    inputs = torch.zeros(9, 6, 30, 30)
    if config == 'classical_only':
        l_stim_type = [0]
    else:
        l_stim_type = [1]

    for stim_type_idx in l_stim_type:

        for stim_idx in range(9):
            H_L4 = define_L4_inputs(stim_type_idx, stim_idx)
            H_LM = define_LM_inputs(stim_type_idx, stim_idx)

            H_L4_reshaped = H_L4.reshape(30, 30)
            H_LM_reshaped = H_LM.reshape(30, 30)

            inputs[stim_idx, -2, :, :] = H_L4_reshaped  # inputs from L4
            inputs[stim_idx, -1, :, :] = H_LM_reshaped  # inputs from LM
    return inputs


def main():
    for fitting_config_idx in range(len(l_fitting_config)):
        fitting_config = l_fitting_config[fitting_config_idx]
        print(fitting_config)

        folder_path = '/Users/kriswu/Projects/spatialrnn_axon/examples/spatialrnn/optimized_files/optimized_files_' + fitting_config
        model_names = [f for f in os.listdir(folder_path) if
                       os.path.isfile(os.path.join(folder_path, f)) and f != '.DS_Store']
        sorted_model_names = sorted(model_names, key=lambda name: float(name.replace('.pt', '')))
        n_selected_models = len(sorted_model_names)

        for model_name_idx in range(n_selected_models):
            model_name = sorted_model_names[model_name_idx]
            print(model_name)
            x, model = setup(model_name, fitting_config)
            inputs = get_inputs()

            response_combined = []

            if config == 'classical_only':
                l_stim_type = [0]
            else:
                l_stim_type = [1]
            for stim_type_idx in l_stim_type: # classical stimulus and inverse stimulus

                for stim_idx in range(9):
                    x['dh'] = inputs[stim_idx, :, :, :]
                    response = model(x, ndim=x.ndim, to_dataframe="pandas") # something is wrong here.
                    response['inverse'] = stim_type_idx != 0
                    response['size'] = 5 + stim_idx * 10

                    intervals = pd.interval_range(start=0, end=132, freq=6, closed="left")
                    response['distance'] = pd.cut(response['distance'], bins=intervals)
                    response_combined.append(response)

            response_total = pd.concat(response_combined)
            response_total["distance"] = response_total["distance"].astype("category")
            response_total["cell_type"] = response_total["cell_type"].astype("category")

            # grid distance, 120 * 4 * 18 = 8640, order are E classical size 5 - 85, PV classical size 5 -85, ...., VIP , classical size 5 -85, E inverse size 5 - 85 ...
            response_grid = response_total.drop(columns=['dV', 'dh', 'distance', 'space_dV'])  # remove columns
            response_grid['distance'] = np.sqrt(response_grid['space[0]'] ** 2 + response_grid['space[1]'] ** 2) # create the distance

            response_grid = response_grid.groupby(['cell_type', 'distance', 'inverse', 'size'], observed=True).mean(numeric_only=True).reset_index() # combine the entries with the same distance

            response_grid = response_grid[~response_grid["cell_type"].isin(["L4", "LM", "X", "Y"])] # remove L4 and LM
            response_grid = response_grid.sort_values(by=["inverse", "cell_type", "size", "distance"]) # TODO: save this one, good for plotting
            response_grid.to_csv('../simulated_activity/' + fitting_config + '_response_grid_' + str(model_name) + '.csv', index=False)

            # grouped distance, 22 * 4 * 18 = 1584, order are E classical size 5 - 85, PV classical size 5 -85, ...., VIP , classical size 5 -85, E inverse size 5 - 85 ...
            response_grouped = response_total.groupby(['cell_type', 'inverse', 'size', 'distance'], observed=True).mean(
                numeric_only=True).reset_index()
            response_grouped = response_grouped[~response_grouped["cell_type"].isin(["L4", "LM", "X", "Y"])]
            response_grouped = response_grouped.sort_values(by=["inverse", "cell_type", "size", "distance"])
            response_grouped = response_grouped.drop(columns=['dV', 'dh', 'space[0]', 'space[1]', 'space_dV'])  # remove columns
            response_grouped = response_grouped['dr']
            response_grouped.to_csv('../simulated_activity/' + fitting_config + '_response_grouped_' + str(model_name) + '.csv', index=False)

if __name__ == "__main__":
    main()
