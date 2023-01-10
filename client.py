import argparse
import socket
from random import randint
from time import sleep

import dill as pickle

import dataset.mnist_dataset as mnist_dataset

MAX_RECV_LEN = 25 * 1024 * 1024  # 25 Mbytes


def waitOnSignal(socket, msg):
    while (socket.recv(MAX_RECV_LEN)).decode() != msg:
        pass


def establishConnection(clientNum, serverPort):
    print(f"Establishing presence of client {clientNum}")

    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    serverAddr = (socket.gethostbyname("localhost"), serverPort)
    clientSocket.connect(serverAddr)

    print(f"Client {clientNum} connected!")

    return clientSocket


def main(args):
    clientSocket = establishConnection(args.id, args.port)

    waitOnSignal(clientSocket, "connection established")

    fullUserData = mnist_dataset.create_data_subset(args.usercount, args.id)

    for currRound in range(args.rounds):
        sleep(randint(2, 8))
        print(f"Client {args.id} generated sufficient data")
        currRoundDataLoader = mnist_dataset.get_round_dataloader(
            currRound, args.rounds, 64, fullUserData
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--usercount", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)