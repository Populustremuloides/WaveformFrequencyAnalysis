import pandas as pd
from Preprocessing import prepData
dataDict = {
    "a":[1,2,3,4,5],
    "b":[5,6,7,8,9],
    "c":[8,7,6,5,4],
    "d":[4,3,2,1,0]
}

df = pd.DataFrame.from_dict(dataDict)

data = prepData(df, False)
print(data)
print(data[0]) # first index is the column
print(data[:,0]) # second index is the row
print(data.shape)
