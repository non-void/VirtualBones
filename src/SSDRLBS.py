import numpy as np
import torch
from psbody.mesh import Mesh


def euler2rotmat(euler):
    # the similar function in pytorch3d.transforms has significant math error compared to this one
    # don't know why
    sx = torch.sin(euler[:, 0]).view((-1, 1))
    sy = torch.sin(euler[:, 1]).view((-1, 1))
    sz = torch.sin(euler[:, 2]).view((-1, 1))
    cx = torch.cos(euler[:, 0]).view((-1, 1))
    cy = torch.cos(euler[:, 1]).view((-1, 1))
    cz = torch.cos(euler[:, 2]).view((-1, 1))

    mat_flat = torch.hstack([cy * cz,
                             sx * sy * cz - sz * cx,
                             sy * cx * cz + sx * sz,
                             sz * cy,
                             sx * sy * sz + cx * cz,
                             sy * sz * cx - sx * cz,
                             -sy,
                             sx * cy,
                             cx * cy])
    return mat_flat.view((-1, 3, 3))


class SSDRLBS:
    def __init__(self, mesh_path, weights_path, trans_off_path, device):
        self.mesh_path = mesh_path
        self.weights_path = weights_path
        self.trans_off_path = trans_off_path
        self.device = device
        self.mesh = Mesh(filename=mesh_path)
        self.v = torch.from_numpy(self.mesh.v).to(device).float()

        self.rest_pose_h = torch.hstack([self.v, torch.ones([self.v.shape[0], 1]).to(device).float()]). \
            reshape((-1, 4, 1))
        self.weights = torch.from_numpy(np.load(weights_path)).to(device).t().float()
        self.trans_off = np.load(trans_off_path)
        self.joint_num = self.weights.shape[1]
        self.bone_num = self.weights.shape[1]
        self.inv_G = torch.zeros((self.joint_num, 4, 4)).to(device)
        for i in range(self.joint_num):
            self.inv_G[i] = torch.eye(4)
            self.inv_G[i, 0, 3] = -self.trans_off[i][0]
            self.inv_G[i, 1, 3] = -self.trans_off[i][1]
            self.inv_G[i, 2, 3] = -self.trans_off[i][2]

    def pose(self, trans, rot_euler):
        rot_mat = euler2rotmat(rot_euler)
        G = torch.zeros((self.joint_num, 4, 4)).to(self.device)
        for i in range(self.joint_num):
            G[i][0:3, 0:3] = rot_mat[i]
            G[i][0:3, 3] = trans[i].t()
            G[i][3, 3] = 1
        for i in range(self.joint_num):
            G[i] = torch.matmul(G[i], self.inv_G[i])
        T = torch.tensordot(self.weights, G, dims=[[1], [0]])
        posed_v = torch.matmul(T, self.rest_pose_h).view((-1, 4))[:, :3]

        return posed_v

    def batch_pose(self, trans, rot_euler, displacement=None):
        tbptt_length = rot_euler.shape[0]
        batch_size = rot_euler.shape[1]
        rot_mat = euler2rotmat(rot_euler.view((-1, 3)))
        trans = trans.view((-1, 3, 1))
        G = torch.cat([rot_mat, trans], dim=2)
        G = torch.cat([G,
                       torch.tile(torch.Tensor([[[0, 0, 0, 1]]]), dims=[G.shape[0], 1, 1]).to(G.device)],
                      dim=1).view((tbptt_length, batch_size, self.bone_num, 4, 4))
        expanded_inv_G = self.inv_G.view((1, 1, self.bone_num, 4, 4)). \
            expand(tbptt_length, batch_size, self.bone_num, 4, 4)
        G = torch.einsum("abcij,abcjk->abcik", G, expanded_inv_G)
        expanded_weights = (self.weights.view((1, 1, self.weights.shape[0], self.weights.shape[1]))). \
            expand(tbptt_length, batch_size, self.weights.shape[0], self.weights.shape[1])
        T = torch.einsum("abij,abjk->abik", expanded_weights, G.view((G.shape[0], G.shape[1], G.shape[2], -1)))
        T = T.view((tbptt_length, batch_size, -1, 4, 4))
        rest_pose_h = torch.hstack([self.v, torch.ones([self.v.shape[0], 1]).to(trans.device).float()]). \
            reshape((-1, 4, 1))
        expanded_h = rest_pose_h.view((1, 1, rest_pose_h.shape[0], 4, 1)). \
            expand((tbptt_length, batch_size, rest_pose_h.shape[0], 4, 1))
        if displacement is not None:
            expanded_h = torch.tile(rest_pose_h.view((1, 1, rest_pose_h.shape[0], 4, 1)),
                                    [tbptt_length, batch_size, 1, 1, 1]).to(displacement.device)
            expanded_h[:, :, :, :3, :] += displacement.view((tbptt_length, batch_size, -1, 3, 1))
        posed_v = torch.einsum("abcij,abcjk->abcik", T, expanded_h)[:, :, :, :3, :]
        posed_v = posed_v[:, :, :, :3, :].view((posed_v.shape[0], posed_v.shape[1], posed_v.shape[2], -1))
        return posed_v


if __name__ == "__main__":
    pass
