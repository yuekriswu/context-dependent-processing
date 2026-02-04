import os

import torch
import numpy as np
import pandas as pd

from niarb import neurons, nn
from niarb.cell_type import CellType
from niarb.nn.modules import frame
import scipy.io as sio

config = 'classical_only'
l_fitting_config = ['final_gaussian_classical_linear_2']

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

def read_rf_parameters_X(stim_type_idx):
    s = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85])
    para_X = torch.zeros(4, 9)
    if stim_type_idx == 0:
        para_X[0, :] = 1/8 * torch.sqrt(torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85]))
        para_X[1, :] = torch.full((9,), 0.0)
        para_X[2, :] = torch.full((9,), 15.0)
        para_X[3, :] = torch.full((9,), 15.0)
    else:
        para_X[0, :] = torch.full((9,), 0.0)
        para_X[1, :] = torch.full((9,), 0.0)
        para_X[2, :] = torch.full((9,), 15.0)
        para_X[3, :] = torch.full((9,), 15.0)
    return para_X

def read_rf_parameters_Y(stim_type_idx, Y_prefactor):
    s = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85])
    para_Y = torch.zeros(4, 9)
    if stim_type_idx == 0:
        para_Y[0, :] = torch.full((9,), 0.0)
        para_Y[1, :] = torch.full((9,), 0.0)
        para_Y[2, :] = torch.full((9,), 15.0)
        para_Y[3, :] = torch.full((9,), 15.0)
    else:
        para_Y[0, :] = Y_prefactor * (90 - s) /10 * 3
        para_Y[1, :] = Y_prefactor * (90 - s) /10 * 2
        para_Y[2, :] = torch.sqrt(torch.tensor([210, 210, 210, 210, 210, 210, 210, 210, 210]))
        para_Y[3, :] = torch.sqrt(torch.tensor([70, 70, 70, 70, 70, 70, 70, 70, 70]))
    return para_Y

def read_rf_parameters_Z(stim_type_idx, Z_power, Z_prefactor):
    s = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85])
    para_Z = torch.zeros(4, 9)
    if stim_type_idx == 0:
        para_Z[0, :] = torch.full((9,), 0.0)
        para_Z[1, :] = torch.full((9,), 0.0)
        para_Z[2, :] = torch.full((9,), 15.0)
        para_Z[3, :] = torch.full((9,), 15.0)
    else:
        para_Z[0, :] = Z_prefactor * s ** Z_power
        para_Z[1, :] = torch.full((9,), 0.0)
        para_Z[2, :] = torch.full((9,), 15.0)
        para_Z[3, :] = torch.full((9,), 15.0)
    return para_Z

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

def define_X_inputs(para_X, stim_idx, dist_mat):
    # rate field related parameters
    sigma2X2_1, sigma2X2_2 = 2.0 * para_X[2, stim_idx] ** 2, 2.0 * para_X[3, stim_idx] ** 2
    R_X = para_X[0, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2X2_1) - para_X[1, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2X2_2)
    H_X = torch.sqrt(torch.tensor(0.5996) + R_X) - torch.sqrt(torch.tensor(0.5996))
    H_X2 = torch.zeros_like(H_X)
    return H_X, H_X2

def define_Y_inputs(para_Y, stim_idx, dist_mat):
    # rate field related parameters
    sigma2Y2_1, sigma2Y2_2 = 2.0 * para_Y[2, stim_idx] ** 2, 2.0 * para_Y[3, stim_idx] ** 2
    R_Y = para_Y[0, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2Y2_1) - para_Y[1, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2Y2_2)
    H_Y = torch.sqrt(torch.tensor(0.5996) + R_Y) - torch.sqrt(torch.tensor(0.5996))
    H_Y2 = torch.zeros_like(H_Y)
    return H_Y, H_Y2

def define_Z_inputs(para_Z, stim_idx, dist_mat):
    # rate field related parameters
    sigma2Z2_1, sigma2Z2_2 = 2.0 * para_Z[2, stim_idx] ** 2, 2.0 * para_Z[3, stim_idx] ** 2
    R_Z = para_Z[0, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2Z2_1) - para_Z[1, stim_idx] * torch.exp(-dist_mat.pow(2) / sigma2Z2_2)
    H_Z = torch.sqrt(torch.tensor(0.5996) + R_Z) - torch.sqrt(torch.tensor(0.5996))
    H_Z2 = torch.zeros_like(H_Z)
    return H_Z, H_Z2

def setup(model_name, fitting_config):
    if fitting_config == 'final_gaussian_classical_linear_2' or fitting_config == 'final_exp_classical' or fitting_config == 'final_gaussian_classical_linear':
        p_ELM = 1
        q_ELM = 0
        sigma_ELM_1 = 15
        sigma_ELM_2 = 1
    elif fitting_config == 'final_diff_of_gaussian_classical_1':
        p_ELM = 9*3
        q_ELM = 3*3
        sigma_ELM_1 = torch.sqrt(torch.tensor(210))
        sigma_ELM_2 = torch.sqrt(torch.tensor(70))
    elif fitting_config == 'final_diff_of_gaussian_classical_2':
        p_ELM = 9*6
        q_ELM = 3*6
        sigma_ELM_1 = torch.sqrt(torch.tensor(210))
        sigma_ELM_2 = torch.sqrt(torch.tensor(70))
    elif fitting_config == 'final_diff_of_exp_classical_1':
        p_ELM = 2.6*3.5
        q_ELM = 1*3.5
        sigma_ELM_1 = torch.tensor(15)
        sigma_ELM_2 = torch.tensor(5)
    elif fitting_config == 'final_diff_of_exp_classical_2':
        p_ELM = 2.6*6
        q_ELM = 1*6
        sigma_ELM_1 = torch.tensor(15)
        sigma_ELM_2 = torch.tensor(5)
    else:
        pass

    file_path = ("/Users/kriswu/Projects/spatialrnn_axon/examples/spatialrnn/optimized_files/optimized_files_") + fitting_config + "/" + model_name

    opti_networks_tensor = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)['gW']
    opti_networks = opti_networks_tensor.detach().cpu().numpy()

    if fitting_config == 'final_gaussian_classical_linear_2':
        # convert to actual connection strength
        opti_networks[0, :] = opti_networks[0, :]
        opti_networks[1, :] = opti_networks[1, :]
        opti_networks[2, :] = opti_networks[2, :]
        opti_networks[3, :] = opti_networks[3, :]
    else:
        alpha_PYR = 2
        alpha_PV = 2
        alpha_SST = 2
        alpha_VIP = 2

        # convert to actual connection strength
        opti_networks[0, :] = opti_networks[0, :] / (alpha_PYR * np.sqrt(0.5996))
        opti_networks[1, :] = opti_networks[1, :] / (alpha_PV * np.sqrt(1.35))
        opti_networks[2, :] = opti_networks[2, :] / (alpha_SST * np.sqrt(0.6818))
        opti_networks[3, :] = opti_networks[3, :] / (alpha_VIP * np.sqrt(0.8437))

    cell_types = [CellType.PYR, CellType.PV, CellType.SST, CellType.VIP, CellType.L4, CellType.LM, CellType.X, CellType.X2]

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

    diff_Gaussian_mask = torch.ones(8, 8, 30, 30)
    diff_Gaussian_mask[0, 5, :, :] = p_ELM

    diff_Gaussian_mask2 = torch.zeros(8, 8, 30, 30)
    diff_Gaussian_mask2[0, 5, :, :] = q_ELM

    if 'gaussian' in fitting_config:
        space_strength_kernel_inst_1 = nn.Gaussian
        space_strength_kernel_inst_2 = nn.Gaussian
    else:
        space_strength_kernel_inst_1 = nn.Laplace
        space_strength_kernel_inst_2 = nn.Laplace

    # define V1 model
    model = nn.V1(
        ["cell_type", "space"],  # connectivity depends on both cell type and space
        cell_types=cell_types,
        f=nn.Match(
            cases={
                "PYR": nn.Compose(nn.Pow(1, tag="p_PYR"), nn.Rectified()),
                "PV": nn.Compose(nn.Pow(1, tag="p_PV"), nn.Rectified()),
                "SST": nn.Compose(nn.Pow(1, tag="p_SST"), nn.Rectified()),
                "VIP": nn.Compose(nn.Pow(1, tag="p_VIP"), nn.Rectified()),
                "L4": nn.Compose(nn.Pow(2, optim=False, bounds=[1.0, 5.0], tag="p_L4"), nn.Rectified()),
                "LM": nn.Compose(nn.Pow(2, optim=False, bounds=[1.0, 5.0], tag="p_LM"), nn.Rectified()),
                "X": nn.Compose(nn.Pow(2, optim=False, bounds=[1.0, 5.0], tag="p_X"), nn.Rectified()),
                "X2": nn.Compose(nn.Pow(2, optim=False, bounds=[1.0, 5.0], tag="p_X2"), nn.Rectified()),
            },
            default=nn.SSN(),
        ),  # SSN nonlinearity
        tau=[1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],  # relative time constants of E, PV, SST, and VIP
        mode="numerical",  # run numerical simulation
        # space_strength_kernel=None,
        space_strength_kernel=space_strength_kernel_inst_1,
        space_strength_kernel2=space_strength_kernel_inst_2,
        diff_Gaussian_mask=diff_Gaussian_mask,
        diff_Gaussian_mask2=diff_Gaussian_mask2,
        vf_symmetry=False,  # allowing to specify different vf for different cell types, vf: baseline voltage/current for different cell types
    )
    state_dict = {
        "gW": torch.tensor(
            [
                [np.abs(opti_networks[0, 0]),
                 np.abs(opti_networks[0, 1]),
                 np.abs(opti_networks[0, 2]),
                 np.abs(opti_networks[0, 3]),
                 np.abs(opti_networks[0, 4]),
                 np.abs(opti_networks[0, 5]),
                 np.abs(opti_networks[0, 6]),
                 np.abs(opti_networks[0, 7])],
                [np.abs(opti_networks[1, 0]),
                 np.abs(opti_networks[1, 1]),
                 np.abs(opti_networks[1, 2]),
                 np.abs(opti_networks[1, 3]),
                 np.abs(opti_networks[1, 4]),
                 np.abs(opti_networks[1, 5]),
                 np.abs(opti_networks[1, 6]),
                 np.abs(opti_networks[1, 7])],
                [np.abs(opti_networks[2, 0]),
                 np.abs(opti_networks[2, 1]),
                 np.abs(opti_networks[2, 2]),
                 np.abs(opti_networks[2, 3]),
                 np.abs(opti_networks[2, 4]),
                 np.abs(opti_networks[2, 5]),
                 np.abs(opti_networks[2, 6]),
                 np.abs(opti_networks[2, 7])],
                [np.abs(opti_networks[3, 0]),
                 np.abs(opti_networks[3, 1]),
                 np.abs(opti_networks[3, 2]),
                 np.abs(opti_networks[3, 3]),
                 np.abs(opti_networks[3, 4]),
                 np.abs(opti_networks[3, 5]),
                 np.abs(opti_networks[3, 6]),
                 np.abs(opti_networks[3, 7])],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00],
            ]
        ),

        "sigma": torch.tensor(
            [
                [7, 5, 7, 5, 7, sigma_ELM_1, 15, 15],
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
                [1, 1, 1, 1, 1, sigma_ELM_2, 1, 1], # the only value that matters is one that is not 1.
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

    model.load_state_dict(state_dict, strict=False)
    return x, model
    # x: input, and model: recurrent model

def get_inputs(x):
    dist_mat = x["distance"][0]

    if config == 'classical_only':
        inputs = torch.zeros(9, 8, 30, 30)
        l_stim_type = [0]
    elif config == 'inverse_only':
        inputs = torch.zeros(9, 8, 30, 30)
        l_stim_type = [1]
    elif config == 'total':
        inputs = torch.zeros(18, 8, 30, 30)
        l_stim_type = [0, 1]
    else:
        pass

    for stim_type_idx in l_stim_type:

        para_X = read_rf_parameters_X(stim_type_idx)
        # para_Y = read_rf_parameters_Y(stim_type_idx, Y_prefactor)
        # para_Z = read_rf_parameters_Z(stim_type_idx, Z_power, Z_prefactor)

        for stim_idx in range(9):
            H_L4 = define_L4_inputs(stim_type_idx, stim_idx)
            H_LM = define_LM_inputs(stim_type_idx, stim_idx)

            H_X, H_X2 = define_X_inputs(para_X, stim_idx, dist_mat)
            # H_Y, H_Y2 = define_Y_inputs(para_Y, stim_idx, dist_mat)
            # H_Z, H_Z2 = define_Y_inputs(para_Z, stim_idx, dist_mat)

            H_L4_reshaped = H_L4.reshape(30, 30)
            H_LM_reshaped = H_LM.reshape(30, 30)
            H_X_reshaped = H_X.reshape(30, 30)
            H_X2_reshaped = H_X2.reshape(30, 30)
            # H_Y_reshaped = H_Y.reshape(30, 30)
            # H_Y2_reshaped = H_Y2.reshape(30, 30)
            # H_Z_reshaped = H_Z.reshape(30, 30)
            # H_Z2_reshaped = H_Z2.reshape(30, 30)

            inputs[stim_type_idx*9+stim_idx, 4, :, :] = H_L4_reshaped  # inputs from L4
            inputs[stim_type_idx*9+stim_idx, 5, :, :] = H_LM_reshaped  # inputs from LM
            inputs[stim_type_idx*9+stim_idx, 6, :, :] = H_X_reshaped  # inputs from X
            inputs[stim_type_idx*9+stim_idx, 7, :, :] = H_X2_reshaped  # inputs from X
            # inputs[stim_type_idx*9+stim_idx, 8, :, :] = H_Y_reshaped  # inputs from Y
            # inputs[stim_type_idx*9+stim_idx, 9, :, :] = H_Y2_reshaped  # inputs from Y
            # inputs[stim_type_idx*9+stim_idx, 10, :, :] = H_Z_reshaped  # inputs from Y
            # inputs[stim_type_idx*9+stim_idx, 11, :, :] = H_Z2_reshaped  # inputs from Y
    return inputs


def main():
    for fitting_config_idx in range(len(l_fitting_config)):
        fitting_config = l_fitting_config[fitting_config_idx]
        print(fitting_config)

        folder_path = '/Users/kriswu/Projects/spatialrnn_axon/examples/spatialrnn/optimized_files/optimized_files_' + fitting_config
        model_names = [f for f in os.listdir(folder_path) if
                       os.path.isfile(os.path.join(folder_path, f)) and f != '.DS_Store']
        sorted_model_names = sorted(model_names, key=lambda name: float(name.replace('.pt', '')))
        # if len(sorted_model_names) > 10:
        #     n_selected_models = 10
        # else:
        n_selected_models = len(sorted_model_names)

        # n_selected_models = 1

        for model_name_idx in range(n_selected_models):
            model_name = sorted_model_names[model_name_idx]
            print(model_name)
            x, model = setup(model_name, fitting_config)
            inputs = get_inputs(x)

            response_combined = []

            if config == 'classical_only':
                l_stim_type = [0]
            elif config == 'inverse_only':
                l_stim_type = [1]
            elif config == 'total':
                l_stim_type = [0, 1]
            else:
                pass

            for stim_type_idx in l_stim_type: # classical stimulus and inverse stimulus

                for stim_idx in range(9):
                    x['dh'] = inputs[stim_type_idx*9+stim_idx, :, :, :]
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

            response_grid = response_grid[~response_grid["cell_type"].isin(["L4", "LM", "X", "X2"])] # remove L4 and LM
            response_grid = response_grid.sort_values(by=["inverse", "cell_type", "size", "distance"]) # TODO: save this one, good for plotting
            response_grid.to_csv('../simulated_activity/' + fitting_config + '_response_grid_' + str(model_name) + '.csv', index=False)

            # grouped distance, 22 * 4 * 18 = 1584, order are E classical size 5 - 85, PV classical size 5 -85, ...., VIP , classical size 5 -85, E inverse size 5 - 85 ...
            response_grouped = response_total.groupby(['cell_type', 'inverse', 'size', 'distance'], observed=True).mean(
                numeric_only=True).reset_index()
            response_grouped = response_grouped[~response_grouped["cell_type"].isin(["L4", "LM", "X", "X2"])]
            response_grouped = response_grouped.sort_values(by=["inverse", "cell_type", "size", "distance"])
            response_grouped = response_grouped.drop(columns=['dV', 'dh', 'space[0]', 'space[1]', 'space_dV'])  # remove columns
            response_grouped = response_grouped['dr']
            response_grouped.to_csv('../simulated_activity/' + fitting_config + '_response_grouped_' + str(model_name) + '.csv', index=False)

if __name__ == "__main__":
    main()
