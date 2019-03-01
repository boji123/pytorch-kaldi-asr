import kaldi_io
import random
import time
from utils import instances_handler
import numpy as np
import torch
#one should zipped the trainning source and target befor handled by batch loader
#batch loader is a iterator, can be call by for loop
class BatchLoader():
    #triples: list of tuple (key, path of utterances, label)
    #feats will be loaded when itering the batch.
    def __init__(self, trainning_triples, batch_size, pre_load = True, print_info = True, mode='drop'):
        self.data = {}
        self.data['key'] = [triples[0] for triples in trainning_triples]
        self.data['src_seq'] = [triples[1] for triples in trainning_triples]
        self.data['tgt_seq'] = [triples[2] for triples in trainning_triples]

        self.batch_size = batch_size
        self.curr_iter = 0
        self.num_batch = int(len(trainning_triples)/batch_size)
        self.pre_load = pre_load
        self.print_info = print_info

        # if mode = drop, the batch loader will drop the data in final iter if it can't form a batch
        # if mode = all, it will return the final iter
        # for training, use drop, and for decoding&testing, use all 
        self.mode = mode
        if mode!='all' and mode !='drop':
            print('[ERROR] mode of BatchLoader can only be [all] or [drop]')
            exit(1)

        #can speed up the training when data batch is small enough
        if self.pre_load:
            self.data['src_seq'] = self.load_batch_data(0, len(trainning_triples))
            self.data['src_seq'], self.data['src_pad_mask'] = instances_handler.pad_to_longest(self.data['src_seq'])
            self.data['tgt_seq'], self.data['tgt_pad_mask'] = instances_handler.pad_to_longest(self.data['tgt_seq'])
            print('[INFO] data preloaded.')

        if self.print_info:
            print('[INFO] loader initialized. data size:{}, batch_size:{}, iter per epoch:{}.'
                .format(len(self.trainning_triples), self.batch_size, self.num_batch))


    def load_batch_data(self, start, end):
        batch = []
        for scripts in self.data['src_seq'][start:end]:
            mat = kaldi_io.read_mat(scripts)
            batch.append(mat)
        return batch


    def __iter__(self):
        self.curr_iter = 0
        if self.pre_load:
            temp = list(zip(self.data['key'], self.data['src_seq'], self.data['src_pad_mask'], self.data['tgt_seq'], self.data['tgt_pad_mask']))
            random.shuffle(temp)
            self.data['key'], self.data['src_seq'], self.data['src_pad_mask'], self.data['tgt_seq'], self.data['tgt_pad_mask'] = zip(*temp)
            self.data['src_seq'] = np.array(self.data['src_seq'])
            self.data['src_pad_mask'] = np.array(self.data['src_pad_mask'])
            self.data['tgt_seq'] = np.array(self.data['tgt_seq'])
            self.data['tgt_pad_mask'] = np.array(self.data['tgt_pad_mask'])
        else:
            temp = list(zip(self.data['key'], self.data['src_seq'], self.data['tgt_seq']))
            random.shuffle(temp)
            self.data['key'], self.data['src_seq'], self.data['tgt_seq'] = zip(*temp)


        if self.print_info:
            print('[INFO] script list is shuffled')
        return self


    def __next__(self):
        if self.curr_iter < self.num_batch:
            start = self.curr_iter * self.batch_size
            end = start + self.batch_size
        elif self.mode == 'all' and self.curr_iter == self.num_batch:
            start = self.curr_iter * self.batch_size
            end = len(self.data['key'])
            if start == end:
                raise StopIteration();
        else:
            raise StopIteration();

        self.curr_iter += 1

        if self.print_info:
            start_time = time.time()

        #should make a data validation here
        if self.pre_load:
            batch = (self.data['key'][start:end],
                    self.data['src_seq'][start:end],
                    self.data['src_pad_mask'][start:end],
                    self.data['tgt_seq'][start:end],
                    self.data['tgt_pad_mask'][start:end])
        else:
            key = self.data['key'][start:end]
            src_seq = self.load_batch_data(start, end)
            tgt_seq = self.data['tgt_seq'][start:end]
            src_seq, src_pad_mask = instances_handler.pad_to_longest(src_seq)
            tgt_seq, tgt_pad_mask = instances_handler.pad_to_longest(tgt_seq)
            batch = (key, src_seq, src_pad_mask, tgt_seq, tgt_pad_mask)

        if self.print_info:
            print('[INFO] iter {}: data loaded. loading cost {:3.2f} seconds'.format(self.curr_iter, time.time() - start_time))
        return batch