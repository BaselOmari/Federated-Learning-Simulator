# Federated Learning Simulator

## Overview
Introduced in ["Communication-Efficient Learning of Deep Networks from Decentralized Data"](https://arxiv.org/abs/1602.05629) by Brendan McMahan et al., Federated Learning  flips the paradigm of machine learning. 

When training a machine learning model based on client data, instead of having the data be sent from the clients to the server, the model is sent from the server to the clients.

This project provides a multi-threaded simulation of the Federated Average Algorithm presented in the paper, using PyTorch and Python.

### In-Depth
The following is a step-by-step overview of the Federated Averaging algorithm presenetd in McMahan et al.:

1. The server initiates a plain machine learning model with given weights

<center><img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/1.png" alt="Server with Plain Model, and Clients" width="275"/></center>

2. The server sends the model to each client

<center><img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/2.png" alt="Server and Clients with Plain Model" width="275"/></center>

3. The clients update their copy of the model with their local dataset
4. The clients send back the model to the server

<center><img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/3.png" alt="Clients send back updated model" width="275"/></center>

5. The server averages all the weights of all the models that it has received back, weighted by the size of local dataset of each client. This average represents the weights of the trained model

<center><img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/4.png" alt="Server aggregates received models" width="275"/></center>

6. Steps 2-5 are then continually repeated when the clients obtain new data


## Results
The framework is tested against the MNIST dataset and compared with a regular baseline.

The Federated Learning simulator achieved 92% accuracy training on the MNIST dataset, which I compared to a standard baseline that achieved 95% accuracy. While performance was compromised, it is very minimal compared to the privacy-preserving advantages that this framework offers.

