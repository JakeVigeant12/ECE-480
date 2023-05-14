import csv
import math

import numpy as np
from matplotlib import pyplot as plt
import sklearn.decomposition as sklDecomp

imageVectors = []
with open('mnist_test.csv', 'r') as csvfile:
    for row in csvfile:
        row = row.split(',')
        add = np.array(row)
        imageVectors.append(add)
imageVectors.sort(key=lambda item: item[0])
for i in range(len(imageVectors)):
    imageVectors[i] = imageVectors[i].astype(float)


# Test plotting an image

def plotVector(inputImage, trim=True):
    if (trim):
        inputImage = inputImage[1:]
    inputImage = inputImage.reshape(28, 28)
    plt.imshow(inputImage, cmap="gray")
    plt.show()


def plotVectors(inputs, rows, cols, title):
    fig = plt.figure(figsize=(rows, cols))
    for j in range(len(inputs) - 1):
        fig.add_subplot(rows, cols, j + 1)
        plt.imshow(inputs[j].reshape(28, 28))
        plt.axis('off')
    plt.savefig(title + ".jpg")
    plt.show()


def getDigit(imageVectors, digit):
    res = []
    for i in range(len(imageVectors)):
        if (digit == math.floor(imageVectors[i][0])):
            # Shave off label
            res.append(imageVectors[i][1:])
    return res


def plotVals(singVals, title):
    plt.figure()
    xAxis = np.arange(1, len(singVals) + 1, 1)
    plt.yscale("log")
    plt.title(title)
    plt.plot(xAxis, singVals)
    plt.savefig(title + ".jpg")
    plt.show()


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


def proj(vector, component):
    return np.multiply(component, ((np.dot(vector, component)) / (magnitude(component)) ** 2))


def PCA(digit, numComps, title, center=True):
    data = np.array(getDigit(imageVectors, digit))
    meanVec = np.mean(data, 0)
    if (center):
        data = data - meanVec
    covar = np.cov(data.T, bias=True)
    eigValues, eigVectors = np.linalg.eigh(covar)
    reselectedVectors = []
    for i in reversed(range(len(eigValues) - numComps - 1, len(eigValues))):
        reselectedVectors.append(eigVectors[:, i])
    # plotVals(np.flip(eigValues[len(eigValues) - 12: len(eigValues)]), title)
    if (center):
        data = data + meanVec
    newData = []
    for i in range(12):
        projectionSum = np.zeros(784)
        for component in reselectedVectors:
            projection = proj(data[i], component)
            projectionSum += projection
        newData.append(projectionSum)
    return newData


p3Images = PCA(9, 3, str(9))
p4Images = PCA(9, 4, str(9), False)
images = getDigit(imageVectors, 9)[:12]

plotInputs = []
plotInputs.extend(images)
plotInputs.extend(p3Images)
plotInputs.extend(p4Images)


def plotVectors(inputs, rows, cols, title):
    fig = plt.figure(figsize=(rows, cols))
    for j in range(len(inputs) - 1):
        fig.add_subplot(rows, cols, j + 1)
        plt.imshow(inputs[j].reshape(28, 28))
        plt.axis('off')
    plt.savefig(title + ".jpg")
    plt.show()


def plotImgRows(inputs, rows, cols, title):
    fig = plt.figure(figsize=(rows, cols))
    for i in range(1, len(inputs)+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(inputs[i-1].reshape(28,28))
        plt.axis('off')
    plt.axis('off')
    plt.savefig(title + ".jpg")
    plt.show()


plotImgRows(plotInputs, 3, 12, str(9))


