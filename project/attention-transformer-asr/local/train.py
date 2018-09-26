#author: baiji
#main trainning procedure for end to end transformoer based asr system
import argparse
from utils import instances_handler
from utils.BatchLoader import BatchLoader
import time
from tqdm import tqdm
import math
import numpy as np
from utils import constants

import torch
import torch.nn as nn
import torch.optim as optim
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from torch.autograd import Variable

def initialize_batch_loader(read_feats_scp_file, read_text_file, read_vocab_file, batch_size):
    utterances = {}
    with open(read_feats_scp_file, encoding='utf-8') as file:
        for line in file:
            (key,rxfile) = line.split()
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
    #begin & end of sequence
    label = instances_handler.add_control_words_index(label)

    #connect the label and utterances
    trainning_triples = []
    for key in utterances:
        if key in label:
            triples = (key, utterances[key], label[key])
            trainning_triples.append(triples)
    print('[INFO] match {} utterance-label pairs.'.format(len(trainning_triples)))

    #in batch loader, trainning feature will be loaded while itering
    #it increase io but decrease memory use
    batch_loader = BatchLoader(trainning_triples, batch_size)
    return batch_loader


def get_performance(crit, pred, goal, smoothing=False, num_class=None):
    '''
    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        goal = goal * (1 - eps) + (1 - goal) * eps / num_class
        raise NotImplementedError
        #seq_logit.view(-1, seq_logit.size(2))
    '''

    #to be done here
    ''' Apply label smoothing if needed '''
    # batch * length * 
    pred = pred.contiguous().view(-1,pred.size()[2])
    goal = goal.contiguous().view(-1)

    loss = crit(pred, goal)

    pred = pred.max(1)[1]
    n_correct = pred.data.eq(goal.data)
    n_correct = n_correct.masked_select(goal.ne(constants.PAD).data).sum()

    return loss, n_correct


def train_epoch(model, batch_loader, crit, optimizer, mode = 'train'):
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        print('[ERROR] invalid epoch mode')
    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(batch_loader, mininterval=2, desc=' ({}) '.format(mode), leave=False):
        # prepare data
        #key = [triples[0] for triples in batch]
        src = [triples[1] for triples in batch]
        tgt = [triples[2] for triples in batch]
        src_seq, src_pad_mask = instances_handler.pad_to_longest(src)
        tgt_seq, tgt_pad_mask = instances_handler.pad_to_longest(tgt)

        src_seq = Variable(torch.FloatTensor(src_seq)) #batch * max length in batch * padded feature dim
        src_pad_mask = Variable(torch.LongTensor(src_pad_mask)) #batch * maxlength in batch * bool mask dim
        tgt_seq = Variable(torch.LongTensor(tgt_seq)) #batch * max length in batch * padded index dim
        tgt_pad_mask = Variable(torch.LongTensor(tgt_pad_mask)) #batch * maxlength in batch * bool mask dim

        goal = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]

        # loading batch will cost about 1.8 second (average speed, under batchsize 512)
        # and padding batch will cost about 0.3 second, thus io is really time-spending

        # forward
        if mode == 'train':
            optimizer.zero_grad()

        pred = model(src_seq, src_pad_mask, tgt_seq, tgt_pad_mask)
        loss, n_correct = get_performance(crit, pred, goal)

        if mode == 'train':
            loss.backward()
            # update parameters
            optimizer.step()
            optimizer.update_learning_rate()

        # note keeping
        n_words = goal.data.ne(constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]
        print('[INFO] loss:{}'.format(loss.data))
    return total_loss/n_total_words, n_total_correct/n_total_words


def train(model, train_data, eval_data, crit, optimizer, opt, model_options):
    for epoch in range(opt.curr_epoch, opt.epoch + 1):
        print('[INFO] trainning epoch {}.'.format(epoch))

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, crit, optimizer)
        print('[INFO]-----(Training)----- ppl: {:7.3f}, accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(math.exp(min(train_loss, 100)), 100*train_accu, (time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = train_epoch(model, eval_data, crit, optimizer, 'eval')
        print('[INFO]-----(Validation)----- ppl: {:7.3f}, accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(math.exp(min(valid_loss, 100)), 100*valid_accu, (time.time()-start)/60))

        checkpoint = {
        'model': model,
        'model_options': model_options,
        'epoch': epoch,
        'train_options': opt}

        valid_accus += [valid_accu]
        model_name = opt.save_model_perfix + '.epoch{}.accu_{:3.2f}.torch'.format(epoch, 100*valid_accu)
        torch.save(checkpoint, model_name)

        model_name = opt.save_model + '.best.accu_{:3.2f}.torch'.format(100*valid_accu)
        if valid_accu >= max(valid_accus):
            torch.save(checkpoint, model_name)
            print('[Info] The best model has been updated.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_feats_scp_file', required=True)
    parser.add_argument('-read_text_file', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_file', required=True)

    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-epoch', type=int, default=10)
    #the epoch of initialized model is 0, after 1 epoch training, epoch is 1
    #if continue trainning, curr_epoch should be model.epoch + 1
    parser.add_argument('-curr_epoch', type=int, default=1)

    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-gpu_device_ids', type=str, default='0')

    parser.add_argument('-save_model_perfix', required=True)
    opt = parser.parse_args()


    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] prepare trainning.')


    train_data = initialize_batch_loader(opt.read_feats_scp_file, opt.read_text_file, opt.read_vocab_file, opt.batch_size)
    eval_data = initialize_batch_loader(opt.read_feats_scp_file, opt.read_text_file, opt.read_vocab_file, opt.batch_size)
    print('[INFO] batch loader is initialized')


    checkpoint = torch.load(opt.load_model_file)
    model = checkpoint['model']
    model_options = checkpoint['model_options']
    print('[INFO] loading model with parameter: {}'.format(model_options))


    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[constants.PAD] = 0
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

    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] trainning start...')
    train(model, train_data, eval_data, crit, optimizer, opt, model_options)


if __name__ == '__main__':
    main()