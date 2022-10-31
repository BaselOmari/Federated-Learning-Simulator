import argparse
import select
import socket

socket.setdefaulttimeout(15)


def getConnections(serverPort):
    serverAddr = (socket.gethostbyname("localhost"), serverPort)
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(serverAddr)
    serverSocket.listen()

    clientPoller = select.poll()
    initialPoller = select.poll()
    initialPoller.register(serverSocket, select.POLLIN)

    while True:
        try:
            (connection, address) = serverSocket.accept()
        except socket.timeout:
            print("Socket Timed Out: finished gathering clients")
            break
        else:
            print(f"Client {address} connected")
            clientPoller.register(connection, select.POLLIN)

    serverSocket.close()

    return clientPoller


def main(args):
    connectionPoller = getConnections(args.port)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
