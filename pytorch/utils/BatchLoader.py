import kaldi_io
import random
import time


#one should zipped the trainning source and target befor handled by batch loader
#batch loader is a iterator, can be call by for loop
class BatchLoader():
    #triples: list of tuple (key, path of utterances, label)
    #feats will be loaded when itering the batch.
    def __init__(self, trainning_triples, batch_size, print_info = True):
        self.trainning_triples = trainning_triples
        self.batch_size = batch_size
        self.curr_iter = 0
        self.num_batch = int(len(trainning_triples)/batch_size)
        self.print_info = print_info

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
            batch = self.load_batch_data(start, end)
            if self.print_info:
                print('[INFO] iter {}: data loaded. loading cost {:3.2f} seconds'.format(self.curr_iter, time.time() - start_time))
            return batch
        else:
            #will ignore the rest of data
            raise StopIteration()