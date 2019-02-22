import argparse
import time
import math
import torch
import train

from utils import instances_handler

def scale_dict(data_dict, factor):
    data_dict = {key: data_dict[key].mul(factor) for key in data_dict}
    return data_dict

def add_dict(data_dict1, factor, data_dict2):
    data_dict = {key: data_dict1[key].add(factor, data_dict2[key]) for key in data_dict1}
    return data_dict

def sum_average_model(model_list):
    num = len(model_list)
    if num == 1:
        print('[Warning] find only one model, but calling model combining.')
        return model_list[0]
    else:
        factor = 1/num
        data_dict = scale_dict(model_list[0].state_dict(), factor)
        for model in model_list[1:]:
            data_dict = add_dict(data_dict, factor, model.state_dict())
        model_list[0].load_state_dict(data_dict)
        return model_list[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_test_dir', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_dir', required=True)
    parser.add_argument('-load_model_file_list', required=True, nargs='+')
    parser.add_argument('-save_model_dir', required=True)
    parser.add_argument('-use_gpu', action='store_true')
    opt = parser.parse_args()
    
    print('[PROCEDURE] combining model with model averaging...')
    models = []
    for file in opt.load_model_file_list:
        checkpoint = torch.load(opt.load_model_dir + '/' + file, map_location=lambda storage, loc: storage)
        train_options = checkpoint['train_options']
        models.append(checkpoint['model'])

    print('[INFO] model loaded')

    print('[INFO] reading test data...')
    batch_size = 96 # dev384/test187
    test_data = train.initialize_batch_loader(opt.read_test_dir + '/feats.scp', opt.read_test_dir + '/text', opt.read_vocab_file, batch_size)
    print('[INFO] batch loader is initialized')

    vocab_size = len(instances_handler.read_vocab(opt.read_vocab_file))
    crit = train.get_criterion(vocab_size)
    if opt.use_gpu:
        crit = crit.cuda()
    print('[INFO] using cross entropy loss.')

#---------------------------------------------------------------------------------------------------------------------
    '''
    for model in models:
        if opt.use_gpu:
            model = model.cuda()
        start = time.time()
        test_loss, test_accu = train.train_epoch(model, test_data, crit, mode = 'eval', use_gpu = opt.use_gpu)
        print('[INFO]-----(evaluating test set)----- ppl: {:7.3f}, accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(math.exp(min(test_loss, 100)), 100*test_accu, (time.time()-start)/60))
    '''
    model = sum_average_model(models)
    if opt.use_gpu:
        model = model.cuda()
    start = time.time()
    test_loss, test_accu = train.train_epoch(model, test_data, crit, mode = 'eval', use_gpu = opt.use_gpu)
    print('[INFO]-----(evaluating combining set)----- ppl: {:7.3f}, accuracy: {:3.2f} %, elapse: {:3.2f} min'
            .format(math.exp(min(test_loss, 100)), 100*test_accu, (time.time()-start)/60))

    model_name = opt.save_model_dir + '/combined.accu{:3.2f}.torch'.format(100*test_accu)
    checkpoint['model'] = model
    torch.save(checkpoint, model_name)

if __name__ == '__main__':
    main()