import kaldi_io
import random
import time

import threading

#one should zipped the trainning source and target befor handled by batch loader
#batch loader is a iterator, can be call by for loop
class BatchLoader():
    #triples: list of tuple (key, path of utterances, label)
    #feats will be loaded when itering the batch.
    def __init__(self, trainning_triples, batch_size, pre_load = True, print_info = True):
        self.trainning_triples = trainning_triples
        self.batch_size = batch_size
        self.curr_iter = 0
        self.num_batch = int(len(trainning_triples)/batch_size)
        self.pre_load = pre_load
        self.print_info = print_info

        #can speed up the training when data batch is small enough
        if self.pre_load:
            self.trainning_triples = self.load_batch_data(0,len(trainning_triples))
            print('[INFO] data preloaded.')

        if self.print_info:
            print('[INFO] loader initialized. data size:{}, batch_size:{}, iter per epoch:{}.'
                .format(len(self.trainning_triples), self.batch_size, self.num_batch))


    def load_batch_data(self, start, end):
        batch = []
        for triples in self.trainning_triples[start:end]:
            mat = kaldi_io.read_mat(triples[1])
            triples = (triples[0], mat, triples[2])
            batch.append(triples)
        return batch
    '''
    #pad instances to the longest one, making instances to same length
    #so then they can be trained parallely
    #should ensure that instances is array of numpy format
    #etc: [array(...), array(...)] -> array[[...],[...]]
    #it can used to pad both 2-d or 1-d array
    def pad_to_longest(instances):
        max_len = max(len(instance) for instance in instances)
        dim = len(instances[0].shape)

        inst_data = []
        pad_masks = []
        for instance in instances:
            pad_length = max_len - len(instance)
            pad_mask = np.ones(len(instance))
            pad_mask = np.pad(pad_mask, (0,pad_length), 'constant', constant_values = (0,constants.PAD))
            pad_masks.append(pad_mask)

            if dim == 1: #usually label
                instance = np.pad(instance, (0,pad_length), 'constant', constant_values = (0,constants.PAD))
            elif dim == 2: #usually feature
                instance = np.pad(instance, ((0,pad_length),(0,0)), 'constant', constant_values = ((0,constants.PAD),(0,0)))
            else:
                print('[ERROR] undefined padding shape')
                exit(0)
            inst_data.append(instance)
        pad_masks = np.array(pad_masks, dtype=int)
        inst_data = np.array(inst_data)
        return inst_data, pad_masks
    '''
    def __iter__(self):
        self.curr_iter = 0
        random.shuffle(self.trainning_triples)

        if self.print_info:
            print('[INFO] script list is shuffled')
        return self


    def __next__(self):
        if self.curr_iter < self.num_batch:
            start = self.curr_iter * self.batch_size
            end = start + self.batch_size
            self.curr_iter += 1

            #should make a data validation here
            start_time = time.time()
            if self.pre_load:
                batch = self.trainning_triples[start:end]
            else:
                batch = self.load_batch_data(start, end)
            if self.print_info:
                print('[INFO] iter {}: data loaded. loading cost {:3.2f} seconds'.format(self.curr_iter, time.time() - start_time))
            return batch
        else:
            #will ignore the rest of data
            raise StopIteration()