from dataset import mnist_dataset
from model.layers import CNN
from model.train import train, test


def base():
    model = CNN()

    trainSet = mnist_dataset.load_dataset(isTrainDataset=True)
    trainLoader = mnist_dataset.get_dataloader(trainSet)

    testSet = mnist_dataset.load_dataset(isTrainDataset=False)
    testLoader = mnist_dataset.get_dataloader(testSet)

    model = train(model, trainLoader)
    testLoss = test(model, testLoader)

    print("Test accuracy: %.2f%%" % (testLoss * 100))


if __name__ == "__main__":
    base()
