import pandas as pd
from Preprocessing import prepData
from Transform import transformData
# user defined variables
pathToDataFrame = ""
kernelLength = 7
miniBatchSize = 1
numInterpolationSteps = 20
timestepRegime = "linear" # "exponential"
cuda = False
pathToOutput = ""
# ******************************************

'''
Assumptions:
- no gaps in the data
- data are arranged where each column is a different time series
- 
'''

def main():

    # read in df
    df = pd.read_csv(pathToDataFrame)
    # prep for analysis
    data = prepData(df, cuda)
    # run the analysis
    transformedData = transformData(data, kernelLength, timestepRegime, numInterpolationSteps, cuda=cuda)

if __name__ == "__main__":
    main()
