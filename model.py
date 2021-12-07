import attn
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_linear = nn.Linear(self.args.in_dim, self.args.hidden_dim)
        self.position_emb = attn.PositionalEmbedding(self.args)
        self.dropout = nn.Dropout(self.args.dropout)
        self.encoder = Encoder(self.args)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.args.seq_out, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.args.seq_out, out_channels=self.args.seq_out, kernel_size=(1, 1),
                               padding=0)
        self.out_lin = nn.Linear(self.args.hidden_dim, 1)

    def forward(self, inputs):
        x = self.in_linear(inputs)
        x = self.dropout(x.transpose(-1, -2) + self.position_emb(x)).transpose(-1, -2)
        x = self.encoder(x)
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = self.out_lin(y)

        return y


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.attn1 = nn.ModuleList()
        self.attn2 = nn.ModuleList()
        self.attn3 = nn.ModuleList()
        for _ in range(self.args.st_layer):
            self.attn1.append(STTransformer(self.args))
            self.attn2.append(STTransformer(self.args))
            self.attn3.append(STTransformer(self.args))
        self.conv_layer1 = ConvLayer(self.args, stride=2)
        self.conv_layer2 = ConvLayer(self.args, stride=2)
        self.conv_layer3 = ConvLayer(self.args, stride=3)

    def forward(self, x):
        for layer in self.attn1:
            x = layer(x)
        x = self.conv_layer1(x)
        for layer in self.attn2:
            x = layer(x)
        x = self.conv_layer2(x)
        for layer in self.attn3:
            x = layer(x)
        x = self.conv_layer3(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, args, stride):
        super(ConvLayer, self).__init__()
        self.args = args
        self.downConv = nn.Conv1d(in_channels=self.args.hidden_dim,
                                  out_channels=self.args.hidden_dim,
                                  kernel_size=(3, 1),
                                  padding=(1, 0),
                                  padding_mode='circular')
        self.norm = nn.BatchNorm2d(self.args.hidden_dim)
        self.activation = nn.ELU()
        self.stride = stride

    def forward(self, x):
        x = self.downConv(x.transpose(-1, 1))  # [B,L,N,D]
        x = self.norm(x)
        x = self.activation(x).transpose(-1, 1)  # [B,L,N,D]
        x = F.max_pool2d(x.permute(0, 2, 1, 3), kernel_size=(self.stride, 1))
        x = x.transpose(1, 2)
        return x


class STTransformer(nn.Module):
    def __init__(self, args):
        super(STTransformer, self).__init__()
        self.args = args
        self.spat_attn = attn.SpatAttnLayer(self.args)
        self.temp_attn = attn.TempAttnLayer(self.args)
        self.dropout = nn.Dropout(self.args.dropout)
        self.norm1 = nn.LayerNorm(self.args.hidden_dim)
        self.norm2 = nn.LayerNorm(self.args.hidden_dim)
        self.linear1 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.linear2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def forward(self, x):
        spat_x = self.spat_attn(x, x, x)
        st_x = self.temp_attn(spat_x, spat_x, spat_x)
        x = x + self.dropout(st_x)
        y = x = self.norm1(x)
        y = self.dropout(F.relu(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)
