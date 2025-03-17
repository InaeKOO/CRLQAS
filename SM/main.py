import argparse
import torch
from sys import argv

class SM(object):
    def __init__(self, conf):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        learning_rate = conf['agent']['learning_rate']
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

    def predict(H, theta, circuit):
        pass

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    parser.add_argument('--epochs', type=int, default=1000, help='Epoch total numbers')
    args = parser.parse_args(argv)
    return args

def train(agent):
    pass

def evaluate_energy(H, theta, circuit):
    return circuit(theta) @ H @ circuit(theta)

if __name__ == '__main__':

    args = get_args(sys.argv[1:])
    device = torch.device(f"cuda:{args.gpu_id}")
    conf = dict()
    agent = SM(conf)
    for epoch in range(args.epochs):
        pred_E = agent.predict(H, theta)
        data_loss = mse(pred_E, true_E)