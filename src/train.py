import json

from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import random_split
from tqdm import trange, tqdm
import torch

from dataset import Dataset
from bert import Bert


PATHS = '../paths.json'


def main():

    torch.manual_seed(28)

    paths = json.load(open(PATHS, 'r'))
    paths = { key: '%s%s' % (paths['home'], paths[key]) for key in paths if key != 'home' }

    # Load and split data
    dataset = Dataset(paths['clean'])
    trnsize = int(0.6 * len(dataset))
    valsize = int(0.2 * len(dataset))
    tstsize = len(dataset) - (trnsize + valsize)
    train, val, test = random_split(dataset, [trnsize, valsize, tstsize]) 

    # Create batch generators
    params = { 'batch_size' : 32, 'shuffle' : True, 'num_workers': 8 }
    tgen = torch.utils.data.DataLoader(train, **params)
    vgen = torch.utils.data.DataLoader(val, **params)

    epochs = 4
    tsteps = len(tgen) * epochs

    # Load model, optimizer and scheduler for optimizer learning rate reduction
    model = Bert()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=tsteps)

    # Train model
    for epoch in trange(epochs):

        # Run forward pass on training batches and update weights 
        trainprog = tqdm(total=len(tgen), leave=True)
        tcorrect, tloss, ccount, lcount = 0, 0, 0, 0
        for X, Y in tgen: 
            correct, loss = model.train_step(X, Y, optimizer, scheduler)
            tcorrect += correct; tloss += loss
            ccount += len(Y); lcount += 1

            trainprog.set_description('Train => Acc: %.8f Loss: %.8f' % (tcorrect/ccount, tloss/lcount))
            trainprog.refresh()
            trainprog.update(1)

        # Run forward pass on evaluation batches and report accuracy
        valprog = tqdm(total=len(vgen), leave=True)
        tcorrect, tloss, ccount = 0, 0, 0
        for X, Y in vgen: 
            correct, _ = model.eval_step(X, Y)
            tcorrect += correct; ccount += len(Y)

            valprog.set_description('Val => Acc: %.8f' % (tcorrect/ccount))
            valprog.refresh()
            valprog.update(1)

    # Save model parameters
    model.save(paths['distilbert'])


if __name__ == '__main__':
    main()
