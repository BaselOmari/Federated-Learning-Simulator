import argparse
import os
import select
import socket
import sys

DIR = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.append(DIR)

import model.layers as layers
import model.train as train


socket.setdefaulttimeout(15)


def sendAllConnections(connectionList: list[socket.socket], msg):
    for connection in connectionList:
        connection.send(msg.encode())


def getConnections(serverPort):
    serverAddr = (socket.gethostbyname("localhost"), serverPort)
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(serverAddr)
    serverSocket.listen()

    connectionList = []

    while True:
        try:
            (connection, address) = serverSocket.accept()
        except socket.timeout:
            print("Socket Timed Out: finished gathering clients")
            break
        else:
            print(f"Client {address} connected")
            connectionList.append(connection)

    serverSocket.close()

    return connectionList


def main(args):
    connectionList = getConnections(args.port)

    cleanModel = layers.CNN()
    trainingInstructions = train.train

    # For number of rounds specified by the args:
    # 1. Send model to the client
    # 2. Client trains using their data
    # 3. Retrieve updated model from clients
    # 4. Aggregate
    # 5. Test against testing set and print results

    # 1. Send model to the client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
