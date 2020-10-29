import linecache

import torch


class Dataset(torch.utils.data.Dataset):


    def __init__(self, filepath):
        """ Initialize dataset

        Args:
            filepath (string): location of dataset
        """

        self.filepath = filepath
        self.size = len(open(filepath).readlines()) - 1


    def __getitem__(self, idx):
        """ Return item at specified index

        Args:
            idx (int): index of required item

        Returns:
            (string): comment at specified index
            (int): label of comment
        """

        line = linecache.getline(self.filepath, idx+1).strip()
        x, y = line[2:].strip('"'), int(line[0]) 
        return x, y

    
    def __len__(self):
        """ Returns size of dataset

        Returns
            (int): size of dataset
        """

        return self.size
