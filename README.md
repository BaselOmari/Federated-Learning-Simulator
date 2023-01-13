# Federated Learning Simulator

### Overview
Introduced in ["Communication-Efficient Learning of Deep Networks from Decentralized Data"](https://arxiv.org/abs/1602.05629) by Brendan McMahan et al., Federated Learning  flips the paradigm of machine learning. 

When training a machine learning model based on client data, instead of having the data be sent from the clients to the server, the model is sent from the server to the clients.

This project provides a multi-threaded simulation of the Federated Average Algorithm presented in the paper, using PyTorch and Python.


### Results
The framework is tested against the MNIST dataset and compared with a regular baseline.

The Federated Learning simulator achieved 92% accuracy training on the MNIST dataset, which I compared to a standard baseline that achieved 95% accuracy. While performance was compromised, it is very minimal compared to the privacy-preserving advantages that this framework offers.

