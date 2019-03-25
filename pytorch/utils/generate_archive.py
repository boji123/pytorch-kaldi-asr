import kaldi_io
import random
import time
from utils import instances_handler
import numpy as np
import torch


#important! size_archive should be multiple of batch size for convenient
def generate_archive(trainning_triples, size_archive, save_perfix):
    num_archive = int(len(trainning_triples) / size_archive) + 1

    for num in range(num_archive):
        start = num * size_archive
        end = num * size_archive + size_archive
        if end > len(trainning_triples):
            end = len(trainning_triples)

        partial_triples = trainning_triples[start:end]

        archive_data = load_data(partial_triples)
        archive_file = save_perfix + '{}.archive'.format(num)
        torch.save(archive_data, archive_file)

    return num_archive


def load_data(trainning_triples):
    data = {}
    data['key'] = [triples[0] for triples in trainning_triples]
    data['src_seq'] = [triples[1] for triples in trainning_triples]
    data['tgt_seq'] = [triples[2] for triples in trainning_triples]

    #load data
    loaded = []
    for scripts in data['src_seq']:
        mat = kaldi_io.read_mat(scripts)
        loaded.append(mat)
    data['src_seq'] = loaded

    data['src_seq'], data['src_pad_mask'] = instances_handler.pad_to_longest(data['src_seq'])
    data['tgt_seq'], data['tgt_pad_mask'] = instances_handler.pad_to_longest(data['tgt_seq'])

    data['src_seq'] = np.array(data['src_seq'])
    data['src_pad_mask'] = np.array(data['src_pad_mask'])
    data['tgt_seq'] = np.array(data['tgt_seq'])
    data['tgt_pad_mask'] = np.array(data['tgt_pad_mask'])

    archive = {
        'key' : data['key'],
        'src_seq' : data['src_seq'],
        'src_pad_mask' : data['src_pad_mask'],
        'tgt_seq' : data['tgt_seq'],
        'tgt_pad_mask' : data['tgt_pad_mask']
    }
    return archive
