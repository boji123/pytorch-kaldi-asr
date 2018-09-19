#author: baiji
#main trainning procedure for end to end transformoer based asr system
import argparse
from utils import instances_handler
from utils.BatchLoader import BatchLoader

import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

def initialize_batch_loader(read_feats_scp_file, read_text_file, read_vocab_file):
    utterances = {}
    with open(read_feats_scp_file, encoding='utf-8') as file:
        for line in file:
            (key,rxfile) = line.split(' ')
            utterances[key] = rxfile
            #the utterance can be read by 
            #mat = kaldi_io.read_mat(rxfile)

    print('[INFO] get {} utterances from {}.'.format(len(utterances), read_feats_scp_file))

    label_text = {}
    with open(read_text_file, encoding='utf-8') as file:
        for line in file:
            data = line.split()
            key = data[0]
            text = data[1:]
            label_text[key] = text
    print('[INFO] get {} labels from {}.'.format(len(label_text), read_text_file))

    label = instances_handler.apply_vocab(label_text, read_vocab_file, 'word2idx')

    #connect the label and utterances
    trainning_triples = []
    for key in utterances:
        if key in label:
            triples = (key, utterances[key], label[key])
            trainning_triples.append(triples)
    print('[INFO] match {} utterance-label pairs.'.format(len(trainning_triples)))

    batch_size = 512
    batch_loader = BatchLoader(trainning_triples, batch_size)
    return batch_loader

def train(model, batch_loader, crit, optimizer, opt):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_feats_scp_file', required=True)
    parser.add_argument('-read_text_file', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_file', required=True)

    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-gpu_device_ids', type=str, default='0')
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    opt = parser.parse_args()


    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] trainning start...')


    batch_loader = initialize_batch_loader(opt.read_feats_scp_file, opt.read_text_file, opt.read_vocab_file)
    print('[INFO] batch loader is initialized')


    checkpoint = torch.load(opt.load_model_file)
    model = checkpoint['model']
    model_options = checkpoint['options']
    print('[INFO] loading model with parameter: {}'.format(model_options))


    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)
    vocab_size = len(torch.load(opt.read_vocab_file))
    crit = get_criterion(vocab_size)
    print('[INFO] using cross entropy loss.')


    optimizer = ScheduledOptim(
        optim.Adam(
            #model.get_trainable_parameters(),
            filter(lambda p: p.requires_grad,model.get_trainable_parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        model_options.d_model, opt.n_warmup_steps)
    print('[INFO] using adam as optimizer.')


    train(model, batch_loader, crit, optimizer, opt)

if __name__ == '__main__':
    main()