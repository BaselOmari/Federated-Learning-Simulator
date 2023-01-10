from dataset import mnist_dataset
from model.layers import CNN
from model.train import train, test, train_test_split


if __name__ == "__main__":
    model = CNN()

    dataset = mnist_dataset.load_full_dataset()
    trainSet, testSet = train_test_split(dataset, 0.8)

    updatedModel = train(model, trainSet)
    testAcc = test(updatedModel, testSet)

    print("Test accuracy: ", testAcc)
