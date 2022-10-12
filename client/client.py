import argparse
import socket

from clientFSM import csm
from clientReceiver import startReceiver
from clientDataGenerator import run

from threading import Thread


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
    serverAddress = formatAddress(args.address)

    print(serverAddress)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--address", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
