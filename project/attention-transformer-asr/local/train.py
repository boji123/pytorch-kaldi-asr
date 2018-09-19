#author: baiji
#main trainning procedure for end to end transformoer based asr system
import argparse
from utils import instances_handler
from utils.BatchLoader import BatchLoader


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


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('-read_feats_scp_file', required=True)
    parser.add_argument('-read_text_file', required=True)
    parser.add_argument('-read_vocab_file', required=True)

    opt = parser.parse_args()

    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] trainning start...')

    batch_loader = initialize_batch_loader(opt.read_feats_scp_file, opt.read_text_file, opt.read_vocab_file)
    for batch in batch_loader:
        print(len(batch))
        exit(0)


if __name__ == '__main__':
    train()