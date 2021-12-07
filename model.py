import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_linear = nn.Linear(self.args.in_dim, self.args.hidden_dim)

    def forward(self, inputs):
        x = self.in_linear(inputs)
        return x
