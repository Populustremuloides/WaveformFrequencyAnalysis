import pandas as pd
from Preprocessing import prepData
from Transform import transformData
from Kernel import defineTriangularKernel
import numpy as np

dataDict = {
    "a":np.random.random(50),
    "b":np.random.random(50),
    "c":np.random.random(50),
    "d":np.random.random(50)
}

df = pd.DataFrame.from_dict(dataDict)

data = prepData(df, False)
# print(data)
# print(data[0]) # first index is the row
# print(data[:,0]) # second index is the column
# print(data.shape)

transformData(data, miniBatchSize = 1)

