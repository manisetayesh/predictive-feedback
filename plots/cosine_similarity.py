import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CosineSimilarity():
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        self.similarities = []

    def _get_flat_weights(self, model):
        return torch.cat([param.data.view(-1) for param in model.parameters() if param.requires_grad])

    def compute_similarity(self): #to be called for each epoch
        w1 = self._get_flat_weights(self.model1)
        w2 = self._get_flat_weights(self.model2)
        cos_sim = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
        self.similarities.append(cos_sim)
        return cos_sim

    def get_similarities(self):
        return self.similarities

    def plot_similarities(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.similarities, marker='o')
        plt.title('Cosine Similarity Between Models Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        plt.show()