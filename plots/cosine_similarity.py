import torch
import os
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mlp import MultiLayerPerceptron
from models.feedback import FAPerceptron
from models.feedback import WMPerceptron

def get_flattened_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters() if p.requires_grad])

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

num_epochs = 10
similarities_bp_fa = []
similarities_bp_bp = []
similarities_bp_wm = []
for epoch in range(1, num_epochs):
    bp_path = os.path.join("saved_models", f"BP_MLP_epoch{epoch}")
    fa_path = os.path.join("saved_models", f"FA_MLP_epoch{epoch}")
    wm_path = os.path.join("saved_models", f"WM_MLP_epoch{epoch}")

    bp_model = torch.load(bp_path, weights_only=False)
    fa_model = torch.load(fa_path, weights_only=False)
    wm_model = torch.load(wm_path, weights_only=False)

    bp_vec = get_flattened_params(bp_model)
    fa_vec = get_flattened_params(fa_model)
    wm_vec = get_flattened_params(wm_model)

    sim_bp_fa = cosine_similarity(bp_vec, fa_vec)
    sim_bp_bp = cosine_similarity(bp_vec, bp_vec) 
    sim_bp_wm = cosine_similarity(bp_vec, wm_vec)

    similarities_bp_fa.append(sim_bp_fa)
    similarities_bp_bp.append(sim_bp_bp)
    similarities_bp_wm.append(sim_bp_wm)

epochs = list(range(1, num_epochs))

plt.figure(figsize=(8, 5))
plt.plot(epochs, similarities_bp_fa, marker='o', label='BP vs FA', color='blue')
plt.plot(epochs, similarities_bp_bp, marker='x', label='BP vs BP', color='green', linestyle='--')
plt.plot(epochs, similarities_bp_wm, marker='o', label='BP vs WM', color='red')
plt.xlabel("Epoch")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Between Models Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
