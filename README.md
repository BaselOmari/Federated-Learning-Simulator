# Federated Learning Simulator
Federated Learning attempts to flip the paradigm of machine learning. When training a machine learning model based on client data, instead of having the data be sent from the clients to the server, the model is sent from the server to the clients. The clients train the model using their local data before sending the updated model back to the server. The server aggregates all models received from all clients to produce a fully trained model.

This concept was introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data" by Brendan McMahan et al. and I was able to simulate the algorithms presented using PyTorch and Python.

I employed knowledge acquired from my Systems Programming and Concurrency course by making the simulator multithreaded. Each client was simulated using a separate thread.

The FL simulator achieved 92% accuracy training on the MNIST dataset, which I compared to a standard baseline that achieved 95% accuracy. While performance was compromised, it is very minimal compared to the privacy-preserving advantages that this framework offers.
