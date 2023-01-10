from dataset import mnist_dataset
from model.layers import CNN
from model.train import train, test


if __name__ == "__main__":
    model = CNN()

    trainSet = mnist_dataset.load_dataset(isTrainDataset=True)
    testSet = mnist_dataset.load_dataset(isTrainDataset=False)

    updatedModel = train(model, trainSet)
    testLoss = test(updatedModel, testSet)

    print("Test loss: ", testLoss)
