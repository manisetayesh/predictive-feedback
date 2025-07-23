import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools

class CosineSimilarity:
    def __init__(self, models, model_names=None):
        self.models = models
        self.model_names = model_names if model_names else [f"Model {i+1}" for i in range(len(models))]
        self.pairs = list(itertools.combinations(range(len(models)), 2))  
        self.history = [] 

    def compute_layerwise_similarity(self):
        epoch_similarities = {}

        for i, j in self.pairs:
            model1 = self.models[i]
            model2 = self.models[j]
            similarities = []

            for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
                if 'weight' in name1 and param1.requires_grad:
                    w1 = param1.detach().flatten()
                    w2 = param2.detach().flatten()

                    cos_sim = F.cosine_similarity(w1, w2, dim=0)
                    similarities.append((name1, cos_sim.item()))

            epoch_similarities[(i, j)] = similarities

        self.history.append(epoch_similarities)

    def plot_latest(self):
        if not self.history:
            print("No similarity data to plot. Call `compute_layerwise_similarity()` first.")
            return

        epoch_data = self.history[-1]
        num_pairs = len(epoch_data)

        plt.figure(figsize=(12, 5 * num_pairs))
        for idx, ((i, j), sim_list) in enumerate(epoch_data.items()):
            layer_names = [name for name, _ in sim_list]
            cos_vals = [val for _, val in sim_list]
            plt.subplot(num_pairs, 1, idx + 1)
            plt.bar(layer_names, cos_vals)
            plt.xticks(rotation=45)
            plt.ylabel("Cosine Similarity")
            plt.title(f"{self.model_names[i]} vs. {self.model_names[j]}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_over_epochs(self, layer_filter=None):
        if not self.history:
            print("No similarity history to plot.")
            return

        for (i, j) in self.pairs:
            layer_dict = {}
            for epoch_data in self.history:
                sim_list = epoch_data[(i, j)]
                for name, val in sim_list:
                    if layer_filter is not None and name not in layer_filter:
                        continue
                    if name not in layer_dict:
                        layer_dict[name] = []
                    layer_dict[name].append(val)

            plt.figure(figsize=(10, 6))
            for name, values in layer_dict.items():
                plt.plot(values, label=name)

            plt.xlabel("Epoch")
            plt.ylabel("Cosine Similarity")
            plt.title(f"{self.model_names[i]} vs. {self.model_names[j]} Over Time")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

