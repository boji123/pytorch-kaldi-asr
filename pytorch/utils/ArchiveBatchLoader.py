import os
import random
import time
from utils import instances_handler
import numpy as np
import torch


#batch loader is a iterator, can be call by for loop
class ArchiveBatchLoader():
    def __init__(self, archive_perfix, num_archive, batch_size, mode='drop'):
        self.archive_perfix = archive_perfix
        self.num_archive = num_archive
        self.batch_size = batch_size
        self.data = {}
        # if mode = drop, the batch loader will drop the data in final iter if it can't form a batch
        # if mode = all, it will return the final iter
        # for training, use drop, and for decoding&testing, use all 
        self.mode = mode
        if mode!='all' and mode !='drop':
            print('[ERROR] mode of BatchLoader can only be [all] or [drop]')
            exit(1)


    def initialize_archive_data(self, archive_file):
        archive_data = torch.load(archive_file)
        self.data['key'] = archive_data['key']
        self.data['src_seq'] = archive_data['src_seq']
        self.data['src_pad_mask'] = archive_data['src_pad_mask']
        self.data['tgt_seq'] = archive_data['tgt_seq']
        self.data['tgt_pad_mask'] = archive_data['tgt_pad_mask']

        temp = list(zip(self.data['key'], self.data['src_seq'], self.data['src_pad_mask'], self.data['tgt_seq'], self.data['tgt_pad_mask']))
        random.shuffle(temp)
        self.data['key'], self.data['src_seq'], self.data['src_pad_mask'], self.data['tgt_seq'], self.data['tgt_pad_mask'] = zip(*temp)


    def __iter__(self):
        self.curr_iter = 0
        self.curr_archive = 0
        self.finish = False
        return self


    def __next__(self):
        if self.finish:
            raise StopIteration();

        if self.curr_iter == 0:
            archive_file = self.archive_perfix + '{}.archive'.format(self.curr_archive)
            self.initialize_archive_data(archive_file)

        start = self.curr_iter * self.batch_size
        end = start + self.batch_size
        self.curr_iter += 1

        #only occurs when archive end
        if end == len(self.data['key']):
            # if final archive end
            if self.curr_archive ==  self.num_archive - 1:
                self.finish = True
            else:
                self.curr_archive += 1
                self.curr_iter = 0

        #only occurs when final archive
        if end > len(self.data['key']):
            if self.mode == 'all':
                end = len(self.data['key'])
                self.finish = True
            elif self.mode == 'drop':
                raise StopIteration();    

        batch = (self.data['key'][start:end],
                self.data['src_seq'][start:end],
                self.data['src_pad_mask'][start:end],
                self.data['tgt_seq'][start:end],
                self.data['tgt_pad_mask'][start:end])

        return batch