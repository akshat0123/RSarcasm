import json

from torch.utils.data import random_split
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
import numpy as np
import torch

from dataset import Dataset


PATHS = '../paths.json'


def main():

    torch.manual_seed(28)

    # Load paths
    paths = json.load(open(PATHS, 'r'))
    paths = { key: '%s%s' % (paths['home'], paths[key]) for key in paths if key != 'home' }

    # Load and split data
    dataset = Dataset(paths['clean'])
    trnsize = int(0.6 * len(dataset))
    valsize = int(0.2 * len(dataset))
    tstsize = len(dataset) - (trnsize + valsize)
    train, val, test = random_split(dataset, [trnsize, valsize, tstsize]) 

    # Create batch generator
    params = { 'batch_size' : 32, 'shuffle' : True, 'num_workers': 8 }
    tgen = torch.utils.data.DataLoader(train, **params)

    # Count number of tokens in each comment
    lens = []
    for X, Y in tqdm(tgen):
        for x in X:
            lens.append(len(x.split(' ')))

    # Plot distribution of token counts
    mean, std, var = np.mean(lens), np.std(lens), np.var(lens)
    fig, axarr = plt.subplots(1, 1, figsize=(10, 10))
    axarr.hist(lens, bins=50)
    axarr.set_xlim([0, 200])
    axarr.set_title('Mean: %.2f Std: %.2f' % (mean, std, var))
    plt.show()


if __name__ == '__main__':
    main()
