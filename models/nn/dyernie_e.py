import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.emb import RandomEmbedding
from utils.torch_utils import to_device


class DyERNIE_E(torch.nn.Module):
    def __init__(self, d, dim, learning_rate, use_cosh=False, dropout=0):
        super(DyERNIE_E, self).__init__()
        self.name = "Euclidean"
        self.learning_rate = learning_rate
        self.dim = dim
        self.use_cosh = use_cosh
        self.dropout = dropout
        self.curvature = to_device(
            torch.tensor(0.0, dtype=torch.double, requires_grad=False)
        )

        r = 6 / np.sqrt(dim)
        self.P = nn.Parameter(
            to_device(
                torch.tensor(
                    np.random.uniform(-r, r, (len(d.relations), dim)),
                    dtype=torch.double,
                    requires_grad=True,
                )
            )
        )
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

        self.initial_E_euc = nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.initial_E_euc.weight.data = to_device(
            torch.tensor(
                np.random.uniform(-r, r, (len(d.entities), dim)), dtype=torch.double
            )
        )
        self.time_emb_v = nn.Parameter(
            to_device(
                torch.tensor(
                    np.random.uniform(-r, r, (len(d.entities), dim)),
                    dtype=torch.double,
                    requires_grad=True,
                )
            )
        )

        ####relation vector
        self.p_euc = nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.p_euc.weight.data = to_device(
            1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double)
        )

    def emb_evolving_vanilla(self, e_idx, times, use_dropout=False):
        init_embd_e = self.initial_E_euc.weight[e_idx]
        linear_velocities = self.time_emb_v[e_idx]

        # #########all velocity vectors are defined in the tangent space, update the embeddings
        emd_linear_temp = linear_velocities * times[:, :, None]  # batch*nneg*dim

        # ##################drift in the tangent space
        new_embds_e = init_embd_e + emd_linear_temp

        if use_dropout:
            new_embds_e = F.dropout(new_embds_e, p=self.dropout, training=self.training)

        return new_embds_e

    def forward(self, u_idx, r_idx, v_idx, t):
        P = self.P[r_idx]
        u_e = self.emb_evolving_vanilla(u_idx, t)
        v = self.emb_evolving_vanilla(v_idx, t)
        p = self.p_euc.weight[r_idx]

        # transforming it by the diagonal relation matrix
        u_m = u_e * P

        # addition
        v_m = v + p

        # compute the distance between two points.
        sqdist = (u_m - v_m).pow(2).sum(dim=-1)
        predictions = -sqdist + self.bs[u_idx] + self.bo[v_idx]
        return predictions
