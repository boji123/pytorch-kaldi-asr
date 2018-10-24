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
        '''
        self.condition = threading.Condition()
        self.thread = None
        self.thread_stop_flag = False
        self.thread_index_start = 0
        self.thread_index_end = 0
        self.thread_next_batch = None
        '''


    def load_batch_data(self, start, end):
        batch = []
        for triples in self.trainning_triples[start:end]:
            mat = kaldi_io.read_mat(triples[1])
            triples = (triples[0], mat, triples[2])
            batch.append(triples)
        return batch

    '''
    def thread_load_batch_data(self):
        while self.thread_stop_flag == False:
            self.condition.acquire()
            if self.thread_next_batch == None:
                if self.print_info:
                    start_time = time.time()

                self.thread_next_batch = self.load_batch_data(self.thread_index_start, self.thread_index_end)

                if self.print_info:
                    print('[INFO] iter {}: data loaded. loading cost {:3.2f} seconds'.format(self.curr_iter, time.time() - start_time))
            self.condition.wait();
    '''

    def __iter__(self):
        self.curr_iter = 0
        random.shuffle(self.trainning_triples)

        if self.print_info:
            print('[INFO] script list is shuffled')
        return self

        '''
        random.shuffle(self.trainning_triples)

        self.thread = threading.Thread(target=self.thread_load_batch_data)
        self.thread_stop_flag = False
        #set the index of initial batch for loading thread
        self.curr_iter = 0
        self.thread_index_start = self.curr_iter * self.batch_size
        self.thread_index_end = self.thread_index_start + self.batch_size
        self.thread_next_batch = None

        self.thread.start()

        if self.print_info:
            print('[INFO] script list is shuffled')
        return self
        '''

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

        '''
        if self.curr_iter < self.num_batch:
            #to ensure the data is exist
            while self.thread_next_batch == None:
                continue
            self.condition.acquire()
            batch = self.thread_next_batch
            #set the index of next batch for loading thread
            self.curr_iter += 1
            self.thread_index_start = self.curr_iter * self.batch_size
            self.thread_index_end = self.thread_index_start + self.batch_size
            self.thread_next_batch = None
            self.condition.notify()
            self.condition.release()
            return batch

        elif self.curr_iter == self.num_batch:
            #to ensure the data is exist
            while self.thread_next_batch == None:
                continue
            self.condition.acquire()
            batch = self.thread_next_batch
            self.thread_next_batch = None
            self.thread_stop_flag = True
            self.condition.notify()
            self.condition.release()
            self.thread.join()
            return batch

        else:
            raise StopIteration()
        '''