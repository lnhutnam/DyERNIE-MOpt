import torch
import torch.nn as nn
import numpy as np

from models.emb import RandomEmbedding
from utils.spherical_utils import (
    sphere_exp_map_c,
    sphere_log_map_c,
    sphere_sqdist,
    sphere_sum_c,
)
from utils.torch_utils import to_device


class DyERNIE_S(torch.nn.Module):
    def __init__(self, d, dim, learning_rate, fixed_c=None):
        super(DyERNIE_S, self).__init__()
        self.name = "Hypersphere"
        self.learning_rate = learning_rate
        self.p = nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.P = nn.Parameter(
            to_device(
                torch.tensor(
                    np.random.uniform(-1, 1, (len(d.relations), dim)),
                    dtype=torch.double,
                    requires_grad=True,
                )
            )
        )  # in the tangent space
        self.bs = nn.Parameter(
            to_device(
                torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)
            )
        )
        self.bo = nn.Parameter(
            to_device(
                torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)
            )
        )
        self.curvature = to_device(
            torch.tensor(fixed_c, dtype=torch.double, requires_grad=False)
        )
        self.initial_E = nn.Embedding(
            len(d.entities), dim, padding_idx=0
        )  # the initial entity embeddings are learned during training.
        self.p.weight.data = to_device(
            1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double)
        )
        self.initial_E.weight.data = to_device(
            1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double)
        )
        self.time_emb_v = nn.Parameter(
            to_device(
                1e-3
                * torch.randn(
                    (len(d.entities), dim), dtype=torch.double, requires_grad=True
                )
            )
        )  # defined in the Euclidian space

    def emb_evolving_vanilla(self, e_idx, times):
        init_embd_p = self.initial_E.weight[e_idx]  # defined in the spherical space
        linear_velocities = self.time_emb_v[e_idx]
        curvature = self.curvature

        # #########all velocity vectors are defined in the tangent space, update embeddings
        tau = times[:, :, None]
        emd_linear_temp = linear_velocities * tau  # shape: batch*nneg*dim

        # ##################project the initial embedding into the tangent space
        init_embd_e = sphere_log_map_c(init_embd_p, curvature)

        # ##################drift in the tangent space
        new_embds_e = init_embd_e + emd_linear_temp  # + emd_season_temp
        new_embds_p = sphere_exp_map_c(new_embds_e, curvature)
        return new_embds_p

    def forward(self, u_idx, r_idx, v_idx, t):
        curvature = self.curvature

        # Dynamic Embeddings
        u = self.emb_evolving_vanilla(u_idx, t)
        v = self.emb_evolving_vanilla(v_idx, t)
        P = self.P[r_idx]
        p = self.p.weight[r_idx]

        # Moebius matrix-vector multiplication
        # map the original subject entity embedding to the tangent space of the Poincaré ball at 0
        u_e = sphere_log_map_c(u, curvature)

        # transforming it by the diagonal relation matrix
        u_P = u_e * P

        # project back to the poincare ball
        u_m = sphere_exp_map_c(u_P, curvature)

        # Moebius addition
        v_m = sphere_sum_c(v, p, curvature)

        # compute the distance between two points on the Poincare ball along a geodesic.
        sqdist = sphere_sqdist(u_m, v_m, curvature)
        predictions = -sqdist + self.bs[u_idx] + self.bo[v_idx]

        return predictions
