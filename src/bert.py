from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tqdm import trange, tqdm
import torch


class Bert:


    def __init__(self, path=None):
        """ Initialize distilbert model

        Args:
            path (string): path to saved model parameters
        """

        self.devicename = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.devicename)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if path is not None: self.init_trained(path)
        else: self.init_pretrained()


    def init_pretrained(self):
        """ Initialize distilbert model that has not been fine tuned
        """

        self.transformer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        for param in self.transformer.base_model.parameters(): param.requires_grad=False
        self.transformer.to(self.device)


    def init_trained(self, path):
        """ Initialize fine tuned distilbert model

        Args:
            path (string): path to saved model parameters
        """

        self.transformer = DistilBertForSequenceClassification.from_pretrained(path)
        self.transformer.to(self.device)


    def parameters(self):
        """ Return model parameters

        Returns:
            (generator): generator for model parameters
        """
        return self.transformer.parameters()


    def forward(self, sentences, labels=None, require_grad=True):
        """ Run forward pass on model

        Args:
            sentences (tuple): tuple of sentence strings as input
            labels (tuple): tuple of labels corresponding to sentences
            require_grad (boolean): whether or not gradient should be kept 

        Returns:
            (dict): dictionary of loss and logits computed after forward pass
        """

        with torch.set_grad_enabled(require_grad):
            tokenized = self.tokenizer(sentences, max_length=32, truncation=True, padding='max_length', return_tensors="pt").to(self.device)
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        if labels is not None: outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        else: outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        return outputs


    def train_step(self, X, Y, optimizer, scheduler=None):
        """ Train model on batch

        Args:
            X (tuple): tuple of sentence strings as input
            Y (tuple): tuple of labels corresponding to sentences
            optimizer (optimizer from transformers.optimization): optimizer for model training
            scheduler (scheduler from torch.optim.lr_scheduler): scheduler for optimizer learning rate

        Returns:
            (float): number of total correct classifications
            (float): loss over batch
        """

        self.transformer.train()
        outputs = self.forward(X, labels=Y)
        loss, logits = outputs.loss, outputs.logits

        self.transformer.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()

        labels = torch.argmax(logits, dim=1)
        correct = torch.sum(labels==Y)

        return correct.item(), loss.item()


    def eval_step(self, X, Y):
        """ Evaluate model on batch

        Args:
            X (tuple): tuple of sentence strings as input
            Y (tuple): tuple of labels corresponding to sentences

        Returns:
            (float): number of total correct classifications
            (float): loss over batch
        """

        self.transformer.eval() 
        outputs = self.forward(X, labels=Y, require_grad=False)
        loss, logits = outputs.loss, outputs.logits

        labels = torch.argmax(logits, dim=1)
        correct = torch.sum(labels==Y)

        return correct.item(), loss.item()


    def predict(self, X):
        """ Predict labels for batch

        Args:
            X (tuple): tuple of sentence strings as input

        Returns:
            (torch.Tensor): prediction labels for batch 
        """

        self.transformer.eval()
        outputs = self.forward(X, require_grad=False)
        logits = outputs.logits
        labels = torch.argmax(logits, dim=1)

        return labels


    def save(self, path):
        """ Save model parameters

        Args:
            path (string): directory where parameters are to be saved
        """

        self.transformer.save_pretrained(path)
