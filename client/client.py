import argparse
import socket

import clientDataset
from clientFSM import csm

from model import layers, train


def formatAddress(addressStr):
    address = addressStr.split(":")
    return (address[0], int(address[1]))


def establishConnection(clientNum, serverAddr):
    print(f"Establishing presence of client {clientNum}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(serverAddr)

    csm.onEvent("connected")
    print(f"Client {clientNum} connected!")

    return sock


def main(args):
    sock = establishConnection(args.id, formatAddress(args.address))

    fullUserData = clientDataset.create_data_subset(args.usercount, args.id)

    for currRound in range(args.rounds):
        currRoundDataloader = clientDataset.get_round_dataloader(
            currRound, args.rounds, 64, fullUserData
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--usercount", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--address", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
