import json

from torch.utils.data import random_split
from transformers import AdamW
from tqdm import trange, tqdm
import torch

from dataset import Dataset
from bert import Bert


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
    params = { 'batch_size' : 32, 'shuffle' : False, 'num_workers': 8 }
    tgen = torch.utils.data.DataLoader(test, **params)

    # Load fine-tuned DistilBert model
    model = Bert(paths['distilbert'])


    # Calculate accuracy of sarcasm prediction on test set
    testprog = tqdm(total=len(tgen), leave=True)
    tcorrect, ccount = 0, 0
    for X, Y in tgen:
        correct, _ = model.eval_step(X, Y)
        tcorrect += correct; ccount += len(Y)

        testprog.set_description('Test => Acc: %.8f' % (tcorrect/ccount))
        testprog.refresh()
        testprog.update(1)



if __name__ == '__main__':
    main()
