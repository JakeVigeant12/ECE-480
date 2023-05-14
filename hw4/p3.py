import numpy as np
import scipy.integrate as integrate
import matplotlib
from scipy.stats import binom
from scipy.stats import beta as bt
import waterfall_chart
import matplotlib.pyplot as plt
from scipy.stats import norm

rtrue = 0.8
alpha = 5
beta = 5


def betaFormPrior(r):
    if(r<0.5):
        return 2
    else:
        return 0


def playGame(numTimes):
    res = []
    for i in range(numTimes):
        for i in range(10):
            draw = np.random.uniform(0, 1)
            if (draw < rtrue):
                res.append(1)
            else:
                res.append(0)
    return res


def dataLiklihood(dataValues, r):
    headCount = 0;
    for coin in dataValues:
        if coin == 1:
            headCount += 1
    return binom.pmf(headCount, len(dataValues), r)


def normalizeFactor(rValues, unscaledOutputs):
    factor = (1 / (integrate.cumulative_trapezoid(unscaledOutputs, rValues, 1e-300)))
    factor = factor[len(factor) - 1]
    return factor


def unnormalized_r_given_data(data, rSweep, priorValues):
    likliHoodVector = []
    for r in rSweep:
        likliHoodVector.append(dataLiklihood(data, r))
    resultPost = []
    for i in range(len(likliHoodVector)):
        resultPost.append(likliHoodVector[i] * priorValues[i])
    return resultPost


def normalized_r_probabilities(rSweep, data, priorValues):
    normalizedOutput = []
    output = unnormalized_r_given_data(data, rSweep, priorValues)
    factor = normalizeFactor(rSweep, output)
    for val in output:
        normalizedOutput.append(val * factor)
    return normalizedOutput


def pWin(numGames):
    pWinResults = []
    winCount = 0;
    for i in range(1, numGames + 1):
        result = playGame(1)
        headCount = 0
        for coin in result:
            if (coin == 1):
                headCount += 1
        if (headCount <= 6):
            winCount += 1
        pWinResults.append(winCount / i)
    return pWinResults


def posteriorEvolution(numGames, rSweep):
    initial = []
    prevIterPost = []
    completePost = []
    priorVals = []
    for i in range(numGames):
        currData = playGame(1)
        prevIterPost = completePost
        if (i == 0):
            for r in rSweep:
                priorVals.append(betaFormPrior(r))
        else:
            priorVals = prevIterPost
        completePost = normalized_r_probabilities(rSweep, currData, priorVals)
        if (i == 0):
            initial = completePost
    completePost = np.array(completePost)
    initial = np.array(initial)
    subtracted_array = np.subtract(completePost, initial)
    subtracted = list(subtracted_array)
    return subtracted

plt.figure(1)
winValues = pWin(10)
numGames = np.arange(0,10,1)
plt.plot(numGames, winValues)
plt.show()

rSweep = np.arange(0, 1.01, 0.01).tolist()
wFallDiff = posteriorEvolution(10, rSweep)
plt.figure(2)
waterfall_chart.plot(rSweep, wFallDiff)
plt.show()
## pass in number of games. for first game, func  = betaPrior. for next games, func = result of last iteration
