import torch.nn as nn
import torch.nn.functional as F

class MLPClassificationHead(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, n_classes)

        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])

        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.output.weight)

        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        out = F.tanh(self.input(x))

        for layer in self.hidden:
            out = F.tanh(layer(out))

        out = self.output(out)

        return out