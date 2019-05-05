"""
Adapted from PyTorch 1.0 Distributed Trainer with Amazon AWS
"""

import time
import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.multiprocessing import Pool, Process
from torch.utils.data import random_split
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
# from torch.utils.data.distributed import DistributedSampler
from dynamic_dataloader import DynamicDistributedSampler as DistributedSampler
from dynamic_dataloader import get_dynamic_loader
# from dynamic_dataparallel import DistributedDataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from amz_loader import DatasetAmazon
from sklearn.metrics import f1_score


class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = torch.sigmoid(output).round().long().data
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class F1_Score(object):
    def __init__(self):
        self.pos = torch.zeros(2)
        self.tp = torch.zeros(2)
        self.fn = torch.zeros(2)
        self.score = torch.zeros(2)

    def update(self, output, label):
        for tag in range(2):
            self.pos[tag] += torch.eq(output, tag).sum().to(dtype = torch.float)
            self.tp[tag] += torch.eq(label[torch.eq(output, tag)],tag).sum().to(dtype = torch.float)
            precision = self.tp[tag] / self.pos[tag]
            self.fn[tag] += torch.eq(label[1-torch.eq(output, tag)],tag).sum().to(dtype = torch.float)
            recall = self.tp[tag] / torch.add(self.tp[tag], self.fn[tag]).to(dtype = torch.float)
            self.score[tag] = 0 if precision == 0 and recall == 0 else 2 * (precision * recall) / (precision + recall) 

    @property
    def f1_score(self):
        return self.score.mean().cpu().data.numpy()
    
    def __str__(self):
        return '{:.2f}%'.format(self.f1_score)


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader, loss, dynamic):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.timer = 0
        self.total_batch = train_loader.batch_size*dist.get_world_size()

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = self.train()
            train_time = time.time() - epoch_start
            print("Train Time: ", train_time)
            test_loss, test_acc, test_f1 = self.evaluate()
            epoch_time = time.time()-epoch_start
            # updating the batch dynamically
            # self.train_loader.sampler.update_load(self.timer, 100)
            if (dynamic == 0 and epoch == 1) or dynamic > 0:
                self.train_loader = get_dynamic_loader(self.train_loader, self.timer, self.total_batch)
            print('Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
                'epoch time: {}'.format(epoch_time))
        torch.save(self.net, '.')

    def train(self):
        train_loss = Average()
        train_f1 = F1_Score()
        begin_time = time.time()

        print("Self.net.train: ", time.time()-begin_time)
        forward_timer = 0
        loss_timer = 0
        backward_timer = 0
        opti_timer = 0
        update_timer = 0
        total_timer = 0
        load_timer = 0
        count = 0
        load_start = time.time()

        i = 0
        self.net.train()
        for data, label in self.train_loader:
            load_timer += time.time()-load_start
            #start_time = time.time()
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # forward is called here
            forward_start = time.time()
            output = self.net(data)
            forward_timer += time.time()-forward_start

            loss_start = time.time()
            loss = self.loss(output, label.float())
            
            self.optimizer.zero_grad()
            
            backward_start = time.time()
            loss_timer += backward_start - loss_start
            loss.backward()
            
            opti_start = time.time()
            backward_timer += opti_start - backward_start
            self.optimizer.step()
            
            update_start = time.time()
            opti_timer += update_start - opti_start
            train_loss.update(loss.item(), data.size(0))
            train_f1.update(output, label)
            update_timer += time.time() - update_start
            # total_timer += time.time() - start_time
            load_start = time.time()

            i += 1
            if i % 1000 == 0:
                print('Iter {}, Train Loss: {}, F1: {}'.format(i+1, train_loss, train_f1))
        self.timer = forward_timer
        print("Forward Time : {}s".format(forward_timer))
        print("Loss", loss_timer, "Backward", backward_timer, "Opti", opti_timer)
        print("Update Time", update_timer)
        print("Load Time", load_timer)
        print("---")
        return train_loss

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()
        test_f1 = F1_Score()

        self.net.eval()
        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                output = self.net(data)
                loss = self.loss(output, label.float())

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)
                test_f1.update(output, label)

        return test_loss, test_acc, test_f1


class RNN(nn.Module):   
    def __init__(self, n_vocab):
        super().__init__()
        self.n_vocab = n_vocab
        self.embedding_size = 100
        self.hidden_size = 32
        self.num_layers = 2
        self.word_embeddings = nn.Embedding(self.n_vocab, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=0.5)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.permute(1,0,2))
        fc1_out = self.fc1(lstm_out[-1])
        fc2_out = self.fc2(self.relu(fc1_out))
        return fc2_out 


def get_dataloader(root, batch_size, workers = 0):
    amazon = DatasetAmazon(root, 2650000)
    train_length = int(0.9 * len(amazon))
    test_length = len(amazon)-train_length
    amz_train, amz_test = random_split(amazon,(train_length,test_length))
    sampler = DistributedSampler(amz_train)
    train_loader = data.DataLoader(amz_train, shuffle=(sampler is None), batch_size=batch_size, \
                        sampler=sampler, num_workers=workers, drop_last=True)
    test_loader = data.DataLoader(amz_test, shuffle=False, batch_size=batch_size, num_workers=workers, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':
    
    initial_time = time.time()
    print("Collect Inputs...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--dir", type=str, default='data.h5')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--n_vocab", type=int, default=1e4)
    parser.add_argument("--dynamic", type=int, default=-1)
    args = parser.parse_args()
    
    # number of vocabulary
    num_vocab = args.n_vocab

    # Batch Size for training and testing
    batch_size = args.batch
    
    # Number of additional worker processes for dataloading
    workers = args.workers

    # Number of epochs to train for
    num_epochs = args.epochs

    # dynmaic step
    # 0 only updates the dataloader dynamically after the 1st epoch
    # if not updates every given argument
    dynamic_step = args.dynamic

    # Starting Learning Rate
    lr = args.lr

    # Distributed backend type
    dist_backend = 'nccl'
    print("Data Directory: {}".format(args.dir))
    print("Batch Size: {}".format(args.batch))
    print("Max Number of Epochs: {}".format(args.epochs))
    print("Initialize Process Group...")

    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend=dist_backend,
                                         init_method='env://')
    torch.multiprocessing.set_start_method('spawn', force=True)


    # Establish Local Rank and set device on this node
    local_rank = args.local_rank
    dp_device_ids = [local_rank]

    print("Initialize Model...")
    # Construct Model
    model = RNN(num_vocab).cuda()

    # Make model DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

    # define loss function (criterion) and optimizer
    weight = torch.FloatTensor([1]).cuda()
    loss = nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    print("Initialize Dataloaders...")
    train_loader, test_loader = get_dataloader(args.dir, batch_size, workers)
    print("Training...")
    trainer = Trainer(model, optimizer, train_loader, test_loader, loss, dynamic_step)
    trainer.fit(num_epochs)

    print("Total time: {:.3f}s".format(time.time()-initial_time))