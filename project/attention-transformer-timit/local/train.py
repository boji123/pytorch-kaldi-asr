#author: baiji
#main trainning procedure for end to end transformoer based asr system
import argparse
import time
import math
from tqdm import tqdm

from utils import instances_handler
from utils.BatchLoader import BatchLoader
from utils import constants

import torch
import torch.nn as nn
import torch.optim as optim
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

from utils.get_gpu import get_available_gpu_ids

def initialize_batch_loader(read_feats_scp_file, read_text_file, read_vocab_file, batch_size, mode='drop'):
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

    #begin & end of sequence
    label_text = instances_handler.add_control_words(label_text)
    label = instances_handler.apply_vocab(label_text, read_vocab_file, 'word2idx')

    #connect the label and utterances
    trainning_triples = []
    for key in utterances:
        if key in label:
            triples = (key, utterances[key], label[key])
            trainning_triples.append(triples)
    print('[INFO] match {} utterance-label pairs.'.format(len(trainning_triples)))

    #in batch loader, trainning feature will be loaded while itering
    #it increase io but decrease memory use
    batch_loader = BatchLoader(trainning_triples, batch_size, pre_load = True, print_info = False, mode=mode)
    return batch_loader


def get_performance(crit, pred, goal, smoothing=True, num_class=None):
    # batch * length * vocab
    pred = pred.contiguous().view(-1,pred.size()[2])
    goal = goal.contiguous().view(-1)

    loss = cal_loss(pred, goal, smoothing)

    pred = pred.max(1)[1]
    n_correct = pred.data.eq(goal.data)
    n_correct = n_correct.masked_select(goal.ne(constants.PAD).data).sum()

    return loss, n_correct


def cal_loss(pred, goal, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    goal = goal.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, goal.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = nn.functional.log_softmax(pred, dim=1)

        non_pad_mask = goal.ne(constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = nn.functional.cross_entropy(pred, goal, ignore_index=constants.PAD, reduction='sum')

    return loss


#for robust training, use this function to generate error sequence, in case error result spreading
#for each word, it will be setted to a random word by probability error_prob
#random_range defind the range of available random words (it's about the vocab size)
def gengerate_sequence_error(sequence, tgt_pad_mask, error_prob=0.05, random_range=[4,51]):
    #get the pos that should replace
    if sequence.is_cuda:
        seq_device = sequence.get_device()
        pos_mask = torch.rand(sequence.size(), device=seq_device).lt(error_prob).mul(tgt_pad_mask).to(sequence.dtype)
        num_mask = torch.randint(random_range[0], random_range[1] + 1, sequence.size(), dtype=sequence.dtype, device=seq_device)
    else:
        pos_mask = torch.rand(sequence.size()).lt(error_prob).mul(tgt_pad_mask).to(sequence.dtype)
        num_mask = torch.randint(random_range[0], random_range[1] + 1, sequence.size(), dtype=sequence.dtype)

    num_mask = num_mask.mul(pos_mask)
    sequence = sequence.mul(1-pos_mask)

    sequence = sequence + num_mask
    return sequence
#for robust training, use this function to replace sequence with pred sequence
#may improve performance when error decoding occurs
def replace_sequence_error(sequence, tgt_pad_mask, pred_sequence, replace_prob=0.1):
    #get the pos that should replac
    if sequence.is_cuda:
        seq_device = sequence.get_device()
        pos_mask = torch.rand(sequence.size(), device=seq_device).lt(replace_prob).mul(tgt_pad_mask).to(sequence.dtype)
    else:
        pos_mask = torch.rand(sequence.size()).lt(replace_prob).mul(tgt_pad_mask).to(sequence.dtype)

    pred_mask = pred_sequence.mul(pos_mask) #confirm position to replace
    sequence[:,1:] = sequence[:,1:].mul(1-pos_mask[:,:-1]) #zero the position to replace

    sequence[:,1:] += pred_mask[:,:-1]
    return sequence

def train_epoch(model, batch_loader, crit, mode = 'train', optimizer = None, batch_eval = 10, use_gpu = False, seq_error_prob = 0):
    if mode == 'train':
        model.train()
        batch_loader.mode = 'drop'
    elif mode == 'eval':
        #batch_eval is setted specially for training set
        #so we don't need to eval the whole set
        batch_eval_count = 0
        model.eval()
        batch_loader.mode = 'all'
    else:
        print('[ERROR] invalid epoch mode')
    total_loss = 0
    n_total_words = 0
    n_total_correct = 0


    #for batch in tqdm(batch_loader, mininterval=2, desc='({})'.format(mode)):
    for batch in batch_loader:
        src_seq = batch[1]
        src_pad_mask = batch[2]
        tgt_seq = batch[3]
        tgt_pad_mask = batch[4]

        src_seq = torch.FloatTensor(src_seq) #batch * max length in batch * padded feature dim
        src_pad_mask = torch.ByteTensor(src_pad_mask) #batch * maxlength in batch * bool mask dim
        # warning: embedding require long tensor, maybe it's a waste of menory, waiting to be solved
        tgt_seq = torch.LongTensor(tgt_seq) #batch * max length in batch * padded index dim
        tgt_pad_mask = torch.ByteTensor(tgt_pad_mask) #batch * maxlength in batch * bool mask dim

        if use_gpu:
            src_seq = src_seq.cuda()
            src_pad_mask = src_pad_mask.cuda()
            tgt_seq = tgt_seq.cuda()
            tgt_pad_mask = tgt_pad_mask.cuda()

        goal = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]

        # loading batch will cost about 1.8 second (average speed, under batchsize 512)
        # and padding batch will cost about 0.3 second, thus io is really time-spending

        # forward
        if mode == 'train':
            optimizer.zero_grad()

        
        use_seq_error = False
        if mode == 'train' and use_seq_error and seq_error_prob > 0:
            seq_random_range = [4,51] #including 4 & 51
            tgt_seq = gengerate_sequence_error(tgt_seq, tgt_pad_mask, seq_error_prob, seq_random_range)
        
        pred = model(src_seq, src_pad_mask, tgt_seq, tgt_pad_mask)

        #-------------------error training procedure---------------------------------------
        use_seq_error = False
        if mode == 'train' and use_seq_error and seq_error_prob > 0:
            pred_sequence = pred.max(2)[1]
            tgt_seq = replace_sequence_error(tgt_seq, tgt_pad_mask, pred_sequence, seq_error_prob)
            pred = model(src_seq, src_pad_mask, tgt_seq, tgt_pad_mask) #replace the pred result
        #----------------------------------------------------------------------------------
        if mode == 'train':
            smoothing = True
        else:
            smoothing = False
        smoothing = False
        loss, n_correct = get_performance(crit, pred, goal, smoothing=smoothing)

        if mode == 'train':
            loss.backward()
            # update parameters
            optimizer.step()
            optimizer.update_learning_rate()

        # note keeping
        n_words = goal.data.ne(constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct

        total_loss += loss.data

        if mode == 'eval':
            batch_eval_count += 1
            if batch_eval_count == batch_eval:
                break

    return float(total_loss)/int(n_total_words), float(n_total_correct)/int(n_total_words)


def train(model, train_data, dev_data, test_data, crit, optimizer, opt, model_options):
    train_start_time = time.time()
    best_epoch = 0
    best_accu = 0
    for epoch in range(1, opt.epoch + 1):
        print('[INFO] trainning epoch {}.'.format(epoch))

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, crit, mode = 'train', optimizer = optimizer, use_gpu = opt.use_gpu, seq_error_prob = opt.seq_error_prob)
        print('[INFO]-----(Training)----- accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(100*train_accu, (time.time()-start)/60))

        #eval the training result(after dropout off)
        start = time.time()
        eval_batch_num = 10
        valid_loss, valid_accu = train_epoch(model, train_data, crit, mode = 'eval', batch_eval = eval_batch_num, use_gpu = opt.use_gpu)
        print('[INFO]-----(evaluating train set for {} batch)----- accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(eval_batch_num, 100*valid_accu, (time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = train_epoch(model, dev_data, crit, mode = 'eval', use_gpu = opt.use_gpu)
        print('[INFO]-----(evaluating dev set)----- accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(100*valid_accu, (time.time()-start)/60))

        if valid_accu > best_accu:
            best_accu = valid_accu
            best_epoch = epoch
            best_model = model

        start = time.time()
        test_loss, test_accu = train_epoch(model, test_data, crit, mode = 'eval', use_gpu = opt.use_gpu)
        print('[INFO]-----(evaluating test set)----- accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(100*test_accu, (time.time()-start)/60))

        #early model will be saved only at each interval, and all of model in the last interval will be keep for model combining
        if epoch % opt.save_interval == 0 or opt.epoch - epoch < opt.save_interval:
            checkpoint = {
                'model': model,
                'model_options': model_options,
                'epoch': epoch,
                'train_options': opt}
            model_name = opt.save_model_dir + '/epoch.{}.torch'.format(epoch)
            torch.save(checkpoint, model_name)
            print('[INFO] checkpoint of epoch {} is saved to {}'.format(epoch, model_name))

    print('[INFO] trainning finish.\n\ttime consume: {:3.2f} minute\n\tbest valid accuracy: {:3.2f} %, on epoch {}'
        .format((time.time()-train_start_time)/60, 100*best_accu, best_epoch))
    checkpoint = {
        'model': best_model,
        'model_options': model_options,
        'epoch': best_epoch,
        'train_options': opt}
    model_name = opt.save_model_dir + '/best.epoch{}.accu{:3.2f}.torch'.format(best_epoch, 100*best_accu)
    torch.save(checkpoint, model_name)
    print('[INFO] best model is saved to {}'.format(model_name))
    return best_accu, best_epoch


#-------------------------------for model combining-------------------------------
def scale_dict(data_dict, factor):
    data_dict = {key: data_dict[key].mul(factor) for key in data_dict}
    return data_dict

def add_dict(data_dict1, factor, data_dict2):
    data_dict = {key: data_dict1[key].add(factor, data_dict2[key]) for key in data_dict1}
    return data_dict

def combine(opt, epoch, crit, data, num_model = 20):
    print('[PROCEDURE] combining model with model averaging...')
    start = epoch
    end = epoch - num_model
    models = []
    for i in range(start, end, -1):
        filename = 'epoch.{}.torch'.format(i)
        checkpoint = torch.load(opt.save_model_dir + '/' + filename, map_location=lambda storage, loc: storage)
        train_options = checkpoint['train_options']
        models.append(checkpoint['model'])
    print('[INFO] model loaded')

    best_accu = 0
    for i in range(num_model):
        print('[INFO] averaging {} models'.format(i+1))
        if i == 0:
            model = models[0]
        else:
            model = model.cpu()
            factor = 1/(i+1)
            curr_data_dict = scale_dict(model.state_dict(), 1 - factor)
            next_data_dict = add_dict(curr_data_dict, factor, models[i].state_dict())
            model.load_state_dict(next_data_dict)

        if opt.use_gpu:
            model = model.cuda()
        start = time.time()
        test_loss, test_accu = train_epoch(model, data, crit, mode = 'eval', use_gpu = opt.use_gpu)
        print('[INFO]-----(evaluating combining set)----- ppl: {:7.3f}, accuracy: {:3.2f} %, elapse: {:3.2f} min'
                .format(math.exp(min(test_loss, 100)), 100*test_accu, (time.time()-start)/60))
        if test_accu > best_accu:
            best_accu = test_accu
            best_model = model

    print('[INFO] best combined model with accuracy: {:3.2f} %'.format(100*best_accu))

    model_name = opt.save_model_dir + '/combined.accu{:3.2f}.torch'.format(100*best_accu)
    checkpoint['model'] = best_model
    torch.save(checkpoint, model_name)
#---------------------------------------------------------------------------------------------------


def get_criterion(vocab_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    return nn.CrossEntropyLoss(weight, reduction='sum')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_train_dir', required=True)
    parser.add_argument('-read_dev_dir', required=True)
    parser.add_argument('-read_test_dir', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_file', required=True)
    parser.add_argument('-save_model_dir', required=True)
    #the epoch of initialized model is 0, after 1 epoch training, epoch is 1
    #if continue trainning, curr_epoch should be model.epoch + 1
    parser.add_argument('-seq_error_prob', type=float, default=0)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-optim_start_lr', type=float, default=0.001)
    parser.add_argument('-optim_soft_coefficient', type=float, default=1000)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('-save_interval', type=int, default=10)
    opt = parser.parse_args()

    if opt.use_gpu:
        available_gpu_ids = get_available_gpu_ids()
        if len(available_gpu_ids) == 0:
            print('[ERROR] no cuda device available!')
            exit(1)
        else:
            torch.cuda.set_device(available_gpu_ids[0])
            print('[INFO] use gpu device {}'.format(available_gpu_ids[0]))

    print('[PROCEDURE] prepare trainning.')


    checkpoint = torch.load(opt.load_model_file, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model_options = checkpoint['model_options']
    print('[INFO] loading model with parameter:\n\t{}'.format(model_options))

    vocab_size = len(instances_handler.read_vocab(opt.read_vocab_file))
    crit = get_criterion(vocab_size)
    print('[INFO] using cross entropy loss.')
    if opt.use_gpu:
        model = model.cuda()
        crit = crit.cuda()

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(),
            betas=(0.9, 0.999), eps=1e-08),
        start_lr = opt.optim_start_lr,
        soft_coefficient = opt.optim_soft_coefficient)
    print('[INFO] using adam as optimizer.')


    print('[INFO] reading training data...')
    train_data = initialize_batch_loader(opt.read_train_dir + '/feats.scp', opt.read_train_dir + '/text', opt.read_vocab_file, opt.batch_size)

    print('[INFO] reading dev data...')
    dev_data = initialize_batch_loader(opt.read_dev_dir + '/feats.scp', opt.read_dev_dir + '/text', opt.read_vocab_file, opt.batch_size)

    print('[INFO] reading test data...')
    test_data = initialize_batch_loader(opt.read_test_dir + '/feats.scp', opt.read_test_dir + '/text', opt.read_vocab_file, opt.batch_size)
    print('[INFO] batch loader is initialized')

    print('[PROCEDURE] trainning start...')
    best_accu, best_epoch = train(model, train_data, dev_data, test_data, crit, optimizer, opt, model_options)

    print('[PROCEDURE] combining start on best epoch {}'.format(best_epoch))
    if opt.epoch > 30:
        num_model = 30
    else:
        num_model = opt.epoch
    best_accu = combine(opt, best_epoch, crit, dev_data, num_model)

if __name__ == '__main__':
    main()
