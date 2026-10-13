import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, dim=2, hidden_layers=(50, 50)):
        super().__init__()
        layers = []
        n_inputs = dim
        for n_neurons in hidden_layers:
            layers.extend([nn.Linear(n_inputs, n_neurons), nn.Tanh()])
            n_inputs = n_neurons
        # Γραμμική έξοδος για τη λύση u(x)
        layers.append(nn.Linear(n_inputs, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINN_Large(PINN):
    def __init__(self):
        super().__init__(dim=2, hidden_layers=(100, 100, 100))


class PINN3D(PINN):
    def __init__(self):
        # Ίδιο δίκτυο με τον τελευταίο, ενεργό ορισμό του αρχικού κώδικα
        super().__init__(dim=3, hidden_layers=(100, 100, 100))
