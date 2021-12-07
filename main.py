import argparse

import numpy as np
import torch
import utils

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

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)


def main():
    args.adj_mx, _ = torch.Tensor(utils.get_adjacency_matrix(args.adj_data, args.num_node))
    dataloader = utils.load_dataset(args.data_file, args.batch_size, args.batch_size, args.batch_size)
    args.scaler = dataloader['scaler']

    print(str(args))
