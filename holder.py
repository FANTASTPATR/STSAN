import torch
import torch.optim as op
import utils
from model import Model


class Holder():
    def __init__(self, args):
        self.args = args
        self.model = Model(self.args).to(self.args.device)
        self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss = utils.masked_mae
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def train(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        targets = targets[:, :self.args.seq_out, :, :]
        prediction = self.args.scaler.inv_transform(outputs)
        loss = self.loss(prediction, targets, 0.0)
        loss.backward()
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
        mape = utils.masked_mape(prediction, targets, 0.0).item()
        rmse = utils.masked_rmse(prediction, targets, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        targets = targets[:, :self.args.seq_out, :, :]
        prediction = self.args.scaler.inv_transform(outputs)
        mae = self.loss(prediction, targets, 0.0).item()
        rmse = utils.masked_rmse(prediction, targets, 0.0).item()
        mape = utils.masked_mape(prediction, targets, 0.0).item()
        return mae, mape, rmse
