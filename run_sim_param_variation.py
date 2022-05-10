import copy
import os
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge

from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MotionNet():
    def __init__(self, ssdr_model, detail_model, ssdrlbs,
                 cloth_pose_mean, cloth_pose_std, cloth_trans_mean, cloth_trans_std, ssdr_res_mean, ssdr_res_std,
                 state):
        self.ssdr_model = ssdr_model
        self.detail_model = detail_model
        self.ssdrlbs = ssdrlbs
        self.cloth_pose_mean = cloth_pose_mean
        self.cloth_pose_std = cloth_pose_std
        self.cloth_trans_mean = cloth_trans_mean
        self.cloth_trans_std = cloth_trans_std
        self.ssdr_res_mean = ssdr_res_mean
        self.ssdr_res_std = ssdr_res_std
        self.state = state
        self.ssdr_hidden = None
        self.detail_hidden = None


def get_results(config_path, anim_path, sim_param, out_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    mlp_dim = config["mlp_dim"]

    joint_list = config["joint_list"]
    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]
    obj_template_path = config["obj_template_path"]
    sim_param_config_path = config["sim_param_config_path"]
    pivot_models = config["pivot_models"]
    pivot_net_dir = config["pivot_net_dir"]
    rbf_mlp_path = config["rbf_mlp_path"]

    garment_template = Mesh_obj(obj_template_path)
    joint_num = len(joint_list)

    with open(sim_param_config_path, "r") as f:
        sim_parms = json.load(f)
        sim_param_arr = np.zeros((len(sim_parms), 3))
        sim_param_keys = [list(sim_parms[i].keys())[0] for i in range(len(sim_parms))]
        for i in range(len(sim_parms)):
            sim_param_arr[i][0] = sim_parms[i][sim_param_keys[i]]["bendstiffness"]
            sim_param_arr[i][1] = sim_parms[i][sim_param_keys[i]]["timescale"]
            sim_param_arr[i][2] = sim_parms[i][sim_param_keys[i]]["density"]
        sim_param_std = np.std(sim_param_arr, axis=0)
        sim_param_mean = np.mean(sim_param_arr, axis=0)
        sim_param_arr_normed = (sim_param_arr - sim_param_mean) / sim_param_std
        sim_param_arr_normed = sim_param_arr_normed[pivot_models]

    motion_networks = []
    for pivot_id in pivot_models:
        sim_param_id = list(sim_parms[pivot_id].items())[0][0]
        ssdrlbs_net_path = os.path.join(pivot_net_dir, sim_param_id, "checkpoints/SSDR.pth.tar")
        detail_net_path = os.path.join(pivot_net_dir, sim_param_id, "checkpoints/SSDRRES.pth.tar")
        ssdrlbs_root_dir = os.path.join(pivot_net_dir, sim_param_id, "{}bones".format(ssdrlbs_bone_num))
        state_path = os.path.join(pivot_net_dir, sim_param_id, "state.npz")

        ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6])
        ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path))
        ssdr_model = ssdr_model.to(device)
        ssdr_model.eval()

        data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                                 face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
        detail_model = GRU_GNN_Model(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * garment_template.v.shape[0]],
                                     10, [3, 8, 16], data.edge_index.to(device))
        detail_model.load_state_dict(torch.load(detail_net_path))
        detail_model.to(device)
        detail_model.eval()

        ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                          os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                          os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                          device)

        state = np.load(state_path)

        cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
        cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
        cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
        cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

        ssdr_res_mean = state["ssdr_res_mean"]
        ssdr_res_std = state["ssdr_res_std"]

        vert_std = state["sim_res_std"]
        vert_mean = state["sim_res_mean"]

        motion_networks.append(MotionNet(ssdr_model, detail_model, ssdrlbs,
                                         cloth_pose_mean, cloth_pose_std, cloth_trans_mean, cloth_trans_std,
                                         ssdr_res_mean, ssdr_res_std, state))

    rbf_mlp_model = MLPModel(mlp_dim)
    rbf_mlp_model.load_state_dict(torch.load(rbf_mlp_path))
    rbf_mlp_model.to(device)
    rbf_mlp_model.eval()

    anim = np.load(anim_path)
    state = motion_networks[0].state
    pose_arr = (anim["poses"] - state["pose_mean"]) / state["pose_std"]
    trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    item_length = pose_arr.shape[0]

    with torch.no_grad():
        for motion_network in motion_networks:
            motion_network.ssdr_hidden = None
            motion_network.detail_hidden = None

        for frame in tqdm(range(item_length)):
            motion_signature = np.zeros(((len(joint_list) + 1) * 3), dtype=np.float32)
            for j in range(len(joint_list)):
                motion_signature[j * 3: j * 3 + 3] = pose_arr[frame, joint_list[j]]
            motion_signature[len(joint_list) * 3:] = trans_arr[frame]

            motion_signature = torch.from_numpy(motion_signature)
            motion_signature = motion_signature.view((1, -1)).to(device)

            final_res_arr = np.zeros((len(pivot_models), garment_template.v.shape[0], 3))
            for index, motion_network in enumerate(motion_networks):
                pred_rot_trans, new_ssdr_hidden = motion_network.ssdr_model(motion_signature,
                                                                            motion_network.ssdr_hidden)
                motion_network.ssdr_hidden = new_ssdr_hidden

                pred_pose = pred_rot_trans.view((-1, 6))[:, 0:3] * motion_network.cloth_pose_std + \
                            motion_network.cloth_pose_mean
                pred_trans = pred_rot_trans.view((-1, 6))[:, 3:6] * motion_network.cloth_trans_std + \
                             motion_network.cloth_trans_mean

                ssdr_res = motion_network.ssdrlbs.batch_pose(
                    pred_trans.reshape((1, 1, pred_trans.shape[0], pred_trans.shape[1])),
                    torch.deg2rad(pred_pose).reshape(
                        (1, 1, pred_pose.shape[0], pred_pose.shape[1])))

                detail_res, new_detail_hidden = motion_network.detail_model(pred_rot_trans, ssdr_res,
                                                                            motion_network.detail_hidden)
                motion_network.detail_hidden = new_detail_hidden

                final_res = ssdr_res.detach().cpu().numpy().reshape((-1, 3)) + \
                            (detail_res.detach().cpu().numpy().reshape(
                                (-1, 3)) * motion_network.ssdr_res_std + motion_network.ssdr_res_mean)
                final_res_arr[index] = final_res

            sim_param_normed = (sim_param - sim_param_mean) / sim_param_std
            sim_param_normed = torch.from_numpy(sim_param_normed).to(device).float().reshape((1, -1))
            pivot_param = torch.from_numpy(sim_param_arr_normed).to(device).float()

            projected_target_param = rbf_mlp_model(sim_param_normed)
            projecetd_pivot_param = rbf_mlp_model(pivot_param)

            projected_param_diff = torch.linalg.norm(projecetd_pivot_param - projected_target_param, dim=1)
            weights = torch.exp(-projected_param_diff / 1.0)
            normed_weights = weights / torch.sum(weights)
            normed_weights = normed_weights.detach().cpu().numpy()
            final_res = np.einsum("i,iab->ab", normed_weights, final_res_arr)

            pose = pose_arr[frame] * state["pose_std"] + state["pose_mean"]
            trans = trans_arr[frame] * state["trans_std"] + state["trans_mean"]

            trans_off = np.array([0,
                                  -2.1519510746002397,
                                  90.4766845703125]) / 100.0
            trans += trans_off

            final_res = np.matmul(Rotation.from_rotvec(pose[0]).as_matrix(),
                                  final_res.transpose()).transpose()
            final_res += trans

            out_obj = copy.deepcopy(garment_template)
            out_obj.v = final_res
            # out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))


if __name__ == "__main__":
    config_path = "assets/dress02_sim_params/config.json"
    anim_path = "anim/anim1.npz"
    out_path = "out"
    device = "cuda:0"
    sim_param = np.array([-8.066074945242537, 0.5042348713899382, 0.07167780009477188])

    get_results(config_path, anim_path, sim_param, out_path, device)
