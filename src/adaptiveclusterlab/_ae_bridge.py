"""Minimal extracted CM-HRL autoencoder bridge.

This module contains only the original ClusteringAE, train_clustering_ae
and get_latent interfaces needed by the experimental controller.
It does not implement the published CM-HRL neural decision policy.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

class ClusteringAE(nn.Module):
    def __init__(self, in_dim, latent=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                     nn.Linear(128,64), nn.ReLU(),
                                     nn.Linear(64,latent))
        self.decoder = nn.Sequential(nn.Linear(latent,64), nn.ReLU(),
                                     nn.Linear(64,128), nn.ReLU(),
                                     nn.Linear(128,in_dim))
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z),z

def train_clustering_ae(X_np, model, centers=None, epochs=12, lr=1e-3, alpha=0.6):
    X_np=np.nan_to_num(X_np,nan=0.0)
    if len(X_np)<4:
        return
    ds=TensorDataset(torch.tensor(X_np,dtype=torch.float32))
    dl=DataLoader(ds,batch_size=min(32,max(4,len(X_np)//2)),shuffle=True,drop_last=True)
    opt=optim.Adam(model.parameters(),lr=lr)
    mse=nn.MSELoss()
    for _ in range(epochs):
        for (xb,) in dl:
            if len(xb)<2:
                continue
            xb=xb.to(device)
            opt.zero_grad()
            recon,z=model(xb)
            loss_rec=mse(recon,xb)
            loss_clus=0.0
            if centers is not None and len(xb)>1:
                dist=torch.cdist(z,centers)
                q=torch.softmax(-dist**2 / 1.0,dim=1)
                p=q**2/(q.sum(0,keepdim=True)+1e-8)
                p=p/(p.sum(1,keepdim=True)+1e-8)
                loss_clus=nn.KLDivLoss(reduction="batchmean")(torch.log(q+1e-8),p)
            loss=(1-alpha)*loss_rec+alpha*loss_clus
            loss.backward()
            opt.step()

def get_latent(model,X_np):
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.tensor(np.nan_to_num(X_np, nan=0.0), dtype=torch.float32).to(device)).cpu().numpy()
        return np.nan_to_num(z, nan=0.0)
