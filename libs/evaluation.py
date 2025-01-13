import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
from braindecode.datautil import load_concat_dataset
from libs.ssl_model import VGGSSL
import viz_helpers
import torch
import torch.nn as nn
import numpy as np

def get_embs(save_dir=None):
    if save_dir and os.path.exists(save_dir + '/embeddings.npy'):
        embs = np.load(save_dir + '/embeddings.npy')
    else:
        windows_ds = load_concat_dataset(path='../data/hbn_preprocessed_windowed_scaled', preload=False)
        window_len_s = 10
        emb_size = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emb = VGGSSL()
        class ContrastiveNet(nn.Module):
            """Contrastive module with linear layer on top of siamese embedder.

            Parameters
            ----------
            emb : nn.Module
                Embedder architecture.
            emb_size : int
                Output size of the embedder.
            dropout : float
                Dropout rate applied to the linear layer of the contrastive module.
            """
            def __init__(self, emb, emb_size, dropout=0.5):
                super().__init__()
                self.emb = emb
                self.pooling = nn.AdaptiveAvgPool2d(32)
                self.clf = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(1024, emb_size),
                    nn.Dropout(dropout),
                    nn.Linear(emb_size, 1)
                )

            def forward(self, x):
                x1, x2 = x
                z1, z2 = self.emb(x1), self.emb(x2)
                z = self.pooling(torch.abs(z1 - z2)).flatten(start_dim=1)

                return self.clf(z).flatten()

        model = ContrastiveNet(emb, emb_size).to(device)
        checkpoint = torch.load('RP_VGGSSL/checkpoints/epoch=7-step=5064.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        dataloader = torch.utils.data.DataLoader(windows_ds, batch_size=1, shuffle=False)
        embs = np.array([])
        for sample in dataloader:
            X = sample[0]
            if embs.shape[0] > 0:
                embs = np.concatenate([embs, model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()])
            else:
                embs = model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()
            # embs = np.concatenate([embs, model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()])
        
        if save_dir:
            np.save(save_dir + '/embeddings.npy', embs)
    return embs

def rankme(embs):
    if len(embs.shape) > 2:
        raise ValueError('Expect 2D embeddings of shape (N, K)')
    if embs.shape[0] < embs.shape[1]:
        raise ValueError('Expect N >= K')
    _, S, _ = np.linalg.svd(embs)
    eps = 1e-7
    p = S/np.linalg.norm(S, ord=1) + eps
    rank_z = np.exp(-np.sum(p*np.log(p)))

    return rank_z