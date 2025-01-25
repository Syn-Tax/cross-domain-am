import torch.nn as nn
import torch.nn.functional as F


class MLPMultilayerClassificationHead(nn.Module):
    """Simple multilayer perceptron for use as classification head
    Variable number of hidden layers and hidden size

    activation function: tanh
    """

    def __init__(
        self,
        input_size,
        n_classes,
        hidden_size=128,
        num_hidden_layers=4,
        activation="relu",
        initialisation="kaiming_normal",
    ):
        super().__init__()

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "sigmoid":
            self.activation = F.sigmoid

        # create input and output layers
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, n_classes)

        # create hidden layers
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )

        # initialise input and output layers
        if initialisation == "kaiming_normal":
            init = nn.init.kaiming_normal_
        elif initialisation == "kaiming_uniform":
            init = nn.init.kaiming_uniform_
        elif initialisation == "uniform":
            init = nn.init.uniform_
        elif initialisation == "normal":
            init = nn.init.normal_
        elif initialisation == "xavier_uniform":
            init = nn.init.xavier_uniform_
        elif initialisation == "xavier_normal":
            init = nn.init.xavier_normal_

        if initialisation != None:
            init(self.input.weight)

            init(self.output.weight)

            # initialise hidden layers
            for layer in self.hidden:
                init(layer.weight)

    def forward(self, x):

        # calculate model outputs
        out = self.activation(self.input(x))

        for layer in self.hidden:
            out = self.activation(layer(out))

        # final layer has no activation function for use with cross entropy loss
        out = self.output(out)

        return out


class MLPClassificationHead(nn.Module):
    """Simple multilayer perceptron for use as classification head

    activation function: tanh
    """

    def __init__(self, input_size, n_classes, *args, **kwargs):
        super().__init__()

        # create input and output layers
        self.input = nn.Linear(input_size, n_classes)

        # initialise input and output layers
        nn.init.kaiming_normal_(self.input.weight)

    def forward(self, x):

        # calculate model outputs
        out = self.input(x)

        return out
