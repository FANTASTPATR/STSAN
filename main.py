import argparse
import time

import numpy as np
import torch
import utils
from holder import Holder
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:7")
parser.add_argument('--data', type=str, default='PEMS-D8', help='dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=500, help='training epoch')
parser.add_argument("--seed", type=int, default=520, help='random seed')
parser.add_argument("--clip", type=float, default=5., help='gradient clip')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay rate')
parser.add_argument("--comment", type=str, default="PEMS-D8-LB1-int5",
                    help='whether recording')
parser.add_argument("--recording", type=bool, default=True, help='whether recording')

parser.add_argument("--num_heads", type=int, default=8, help='heads (GAT)')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument("--seq_in", type=int, default=12, help='historical length')
parser.add_argument("--seq_out", type=int, default=12, help='prediction length')

args = parser.parse_args()

if args.data == "PEMS-D4":
    args.data_file = "./data/PEMS-D4"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"


elif args.data == "PEMS-D8":
    args.data_file = "./data/PEMS-D8"
    args.adj_data = "./data/sensor_graph/distance_pemsd8.csv"
    args.num_node = 170
    args.in_dim = 1
    args.task = "flow"

if args.recording:
    utils.record_info(str(args), "./records/" + args.comment)
    utils.record_info("---",
                      "./records/" + args.comment)
    sw = SummaryWriter(comment=args.comment)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)


def main():
    args.adj_mx, _ = torch.Tensor(utils.get_adjacency_matrix(args.adj_data, args.num_node))
    dataloader = utils.load_dataset(args.data_file, args.batch_size, args.batch_size, args.batch_size)
    args.scaler = dataloader['scaler']

    print(str(args))
    engine = Holder(args)
    print("start training...")

    his_loss = []
    val_time = []
    train_time = []

    for epoch_num in range(args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainX = torch.Tensor(x).to(args.device)
            trainy = torch.Tensor(y).to(args.device)

            metrics = engine.train(trainX, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % 200 == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                if args.recording:
                    utils.record_info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),
                                      "./records/" + args.comment)

        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        print("eval...")
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            valx = torch.Tensor(x).to(args.device)
            valy = torch.Tensor(y).to(args.device)
            metrics = engine.eval(valx[..., 0:1], valy[..., 0:1])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch_num, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        if args.recording:
            sw.add_scalar('Loss/train', mtrain_loss, global_step=epoch_num)
            sw.add_scalar('Loss/valid', valid_loss, global_step=epoch_num)
            sw.add_scalar('MAPE/train', mtrain_mape, global_step=epoch_num)
            sw.add_scalar('MAPE/valid', valid_mape, global_step=epoch_num)
            sw.add_scalar('RMSE/train', mtrain_rmse, global_step=epoch_num)
            sw.add_scalar('RMSE/valid', valid_rmse, global_step=epoch_num)
