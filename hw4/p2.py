import numpy as np
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

variance = np.sqrt(0.5)
def uniformPrior(theta):
    if(theta <= 5 and theta >= 0):
        return (1/5)
    else:
        return 0

def gaussianPrior(theta):
    return norm.pdf(theta,2,np.sqrt(0.2))

def genData(theta, timeList):
    dataPoints = []
    for i in range(len(timeList)):
        xt = np.sin(theta * timeList[i])
        nt = np.random.normal(xt,variance)
        dataPoints.append(xt+nt)
    return dataPoints

def normalizePosterior(outputs, thetaValues):
    factor = (1/(integrate.cumulative_trapezoid(outputs, thetaValues, 1e-300)))
    factor = factor[len(factor)-1]
    return factor

def dataLiklihood(dataValues, times, theta):
    toBeMultiplied = []
    for i in range(len(dataValues)):
        mean = np.sin(theta*times[i])
        current = norm.pdf(dataValues[i],mean,variance)
        toBeMultiplied.append(current)
    return np.prod(toBeMultiplied)

def thetaConditionedOnData(dataValues, times, theta):
    return (dataLiklihood(dataValues, times ,theta)*gaussianPrior(theta))

def evaluatePosterior(thetaSweep, times, dataValues):
    #Range theta and then evaluate the posterior at each value with our data given
    output = []
    for theta in thetaSweep:
        output.append(thetaConditionedOnData(dataValues,times, theta))
    factor = normalizePosterior(output,thetaSweep)
    normOutputs = []
    for val in output:
        normOutputs.append(val*factor)
    return normOutputs





times = np.linspace(0, 2*np.pi, 100)
thetaSweep = list(np.arange(0,5,0.02))
#Hardcoded in true value of theta
dataSamples = genData(3,times)
posteriorValues = evaluatePosterior(thetaSweep, times,dataSamples)

testNorm = ((integrate.cumulative_trapezoid(posteriorValues, thetaSweep, 1e-300)))
print(testNorm[len(testNorm) - 1])

matplotlib.use('TkAgg')
plt.figure(1)
plt.plot(times, dataSamples)
plt.figure(2)
plt.plot(thetaSweep, posteriorValues)
plt.show()