import pandas as pd
import torch.nn.functional as F
import torch
import math
from Kernel import defineTriangularKernel


def getLinearTimesteps(numRows, kernelLength):
    return list(range(1,numRows - kernelLength))

def smoothDataTensor(data, timestep, cuda):
    data = data.unsqueeze(1)

    kernel = torch.ones(timestep)
    kernel = kernel.unsqueeze(0)
    kernel = kernel.unsqueeze(0)
    if cuda:
        kernel = kernel.cuda()
        twoDTensor = data.cuda()
    out = F.conv1d(data, kernel)
    out = out / timestep
    return out


def convolveTensor(miniBatch, kernel):
    if len(miniBatch.shape) == 4: # FIXME: this is the coding equivalent of duct tape :)
        miniBatch = miniBatch.squeeze(1)
    return F.conv1d(miniBatch, kernel)

def getMiniBatch(miniBatchSize, i, numCols, data):
    startCol = i * miniBatchSize
    endCol = startCol + miniBatchSize
    if endCol > numCols:
        endCol == numCols
    colData = data[startCol:endCol]  # select a column
    colData = torch.unsqueeze(colData, -2)  # prep the column for smoothing

    return colData


def transformData(data, kernelLength=10, timestepRegime="linear", miniBatchSize=1, numInterpolationSteps=9, cuda=False):
    assert timestepRegime == "linear" or timestepRegime == "exponential", "timestepRegime must be \"linear\" or \"exponential\""
    assert numInterpolationSteps < kernelLength, "kernel length must be gerater than numInterpolationSTeps"
    numCols = data.shape[0]
    numRows = data.shape[1]

    if timestepRegime == "linear":
        timesteps = getLinearTimesteps(numRows, kernelLength)
    else:
        print("other timestep regimes than linear not implemented")

    for timestep in timesteps:

        numMiniBatches = math.ceil(numCols / miniBatchSize)
        timeStepKernelLength = kernelLength # FIXME: maybe adjust this so that it moves with the timestep
        smoothData = smoothDataTensor(data, timestep, cuda) # smooth according to the timestep

        for j in range(numInterpolationSteps + 1):
            kernel = defineTriangularKernel(timeStepKernelLength, j / numInterpolationSteps)

            if cuda:
                kernel = kernel.cuda()

            # convolve this kernel across every smoothed-over time series
            for i in range(numMiniBatches):
                miniBatch = getMiniBatch(miniBatchSize,i, numCols, smoothData)
                convolvedData = convolveTensor(miniBatch, kernel)

                # store the output

