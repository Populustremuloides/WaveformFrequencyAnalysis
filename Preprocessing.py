import pandas as pd
import torch

def prepData(df, cuda):
    # extract the matrix
    data = df.to_numpy()
    data = data.transpose()
    data = torch.from_numpy(data).float()

    if cuda: # convert to cuda if available
        data = data.cuda()

    return data
