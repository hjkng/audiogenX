import torch
from torch import nn
import torch.nn.functional as F
import random


class MaskGenerator(nn.Module):
    def __init__(self,
                 len_token,
                 dim_token,
                 tau=1.0):
        super(MaskGenerator, self).__init__()

        # random.seed(42)
        # np.random.seed(42)
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)

        self.nfeat = dim_token  # Number of features in the token (T5 embedding dimension : 1536)
        self.len_token = len_token  # Length of the token
        self.tau = tau  # Temperature for gumbel softmax
        self.hidden_dim = 512  # Dimension of the feedforward network model in nn.TransformerDecoder
        self.hard = True

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim_token, self.hidden_dim),
            nn.PReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid())

    def forward(self, x): # x (Text token, Embedding Dim)
        x = self.mlp(x) # MLP
        reparameterized_mask = self.reparameterize(x)
        return x, reparameterized_mask

    def reparameterize(self, mask):
        inv_mask = 1 - mask
        total_mask_prob = torch.stack([inv_mask, mask], dim=-1)
        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau=self.tau, hard=self.hard)
        return total_mask_reparameterize[..., 1]
