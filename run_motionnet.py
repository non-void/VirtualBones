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


def get_results(config_path, anim_path, out_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    joint_list = config["joint_list"]
    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]

    ssdrlbs_root_dir = config["ssdrlbs_root_dir"]
    ssdrlbs_net_path = config["ssdrlbs_net_path"]
    detail_net_path = config["detail_net_path"]
    state_path = config["state_path"]
    obj_template_path = config["obj_template_path"]

    garment_template = Mesh_obj(obj_template_path)

    joint_num = len(joint_list)
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

    anim = np.load(anim_path)
    pose_arr = (anim["poses"] - state["pose_mean"]) / state["pose_std"]
    trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    item_length = pose_arr.shape[0]

    ssdr_hidden = None
    detail_hidden = None

    with torch.no_grad():
        for frame in tqdm(range(item_length)):
            motion_signature = np.zeros(((len(joint_list) + 1) * 3), dtype=np.float32)
            for j in range(len(joint_list)):
                motion_signature[j * 3: j * 3 + 3] = pose_arr[frame, joint_list[j]]
            motion_signature[len(joint_list) * 3:] = trans_arr[frame]

            motion_signature = torch.from_numpy(motion_signature)
            motion_signature = motion_signature.view((1, -1)).to(device)

            pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
            ssdr_hidden = new_ssdr_hidden

            pred_pose = pred_rot_trans.view((-1, 6))[:, 0:3] * cloth_pose_std + \
                        cloth_pose_mean
            pred_trans = pred_rot_trans.view((-1, 6))[:, 3:6] * cloth_trans_std + \
                         cloth_trans_mean

            ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((1, 1, pred_trans.shape[0], pred_trans.shape[1])),
                                          torch.deg2rad(pred_pose).reshape(
                                              (1, 1, pred_pose.shape[0], pred_pose.shape[1])))

            detail_res, new_detail_hidden = detail_model(pred_rot_trans, ssdr_res, detail_hidden)
            detail_hidden = new_detail_hidden

            final_res = ssdr_res.detach().cpu().numpy().reshape((-1, 3)) + \
                        (detail_res.detach().cpu().numpy().reshape((-1, 3)) * ssdr_res_std + ssdr_res_mean)

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
            out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))


if __name__ == "__main__":
    config_path = "assets/dress03/config.json"
    anim_path = "anim/anim3.npz"
    out_path = "out"
    device = "cpu"

    get_results(config_path, anim_path, out_path, device)
