''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants
import math

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_insts=None, tgt_insts=None,
            cuda=True, batch_size=64, shuffle=True, test=False):

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts:
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(
                torch.LongTensor(inst_position), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]
            src_data, src_pos = pad_to_longest(src_insts)

            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                return (src_data, src_pos), (tgt_data, tgt_pos)

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()

            
class DataLoader_transfer(DataLoader):
    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(
                torch.LongTensor(inst_position), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]
            src_data, src_pos = pad_to_longest(src_insts)

            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                inst_data_tensor = Variable(torch.LongTensor(tgt_insts), volatile=self.test)
                inst_data_tensor = inst_data_tensor.cuda()
                return (src_data, src_pos), inst_data_tensor

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
            
            
class Cross_Validation_Datahandler:
    def __init__(self, data, opt, corss_num_k=10, shuffle=True):
        self.opt = opt
        self.dict = data['dict']
        self.data = data['data']
        
        assert self.dict
        assert self.data
        
        print('[info] Data loaded, src dict size is {} , tgt dict size is {} , insts size is {}'
              .format(len(self.dict['src']),len(self.dict['tgt']),len(self.data['src'])))
        
        if shuffle:
            paired_insts = list(zip(self.data['src'], self.data['tgt']))
            random.shuffle(paired_insts)
            self.data['src'], self.data['tgt'] = zip(*paired_insts)
            print('[info] Data shuffled')

        assert len(self.data['src']) == len(self.data['tgt'])
        self.data_size = len(self.data['src'])
        self.corss_num_k = corss_num_k
        self.cross_size = math.floor(self.data_size/self.corss_num_k)
        
        #dropped_num = self.data_size - self.cross_size * self.corss_num_k 
        #print('{} of data is divided to {} part, size of each part is {}, with {} dropped'.format(self.data_size, self.corss_num_k, dropped_num))
        
    def load_data(self, epoch_index):
        validation_index = epoch_index % self.corss_num_k
        valid_begin = validation_index * self.cross_size
        valid_end = (validation_index + 1) * self.cross_size
        
        valid_src = self.data['src'][valid_begin:valid_end]
        valid_tgt = self.data['tgt'][valid_begin:valid_end]
        
        #print(valid_src[0])
        #print(valid_tgt[0])
        #exit(0)
        
        train_src = self.data['src'][:valid_begin] + self.data['src'][valid_end:]
        train_tgt = self.data['tgt'][:valid_begin] + self.data['tgt'][valid_end:]
        
        training_data = DataLoader(
            self.dict['src'],
            self.dict['tgt'],
            src_insts=train_src,
            tgt_insts=train_tgt,
            batch_size=self.opt.batch_size,
            shuffle=False,
            cuda=self.opt.cuda)
        validation_data = DataLoader(
            self.dict['src'],
            self.dict['tgt'],
            src_insts=valid_src,
            tgt_insts=valid_tgt,
            batch_size=self.opt.batch_size,
            shuffle=False,
            test=True,
            cuda=self.opt.cuda)
        print('[Data] Using k:{} as valid, range is {}-{}'.format(validation_index, valid_begin, valid_end))
        return training_data, validation_data