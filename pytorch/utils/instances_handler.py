#handling the instance text, generete a dict and return the instance
import torch
from utils import constants
import numpy as np

# read the instance into a {key:value} dict
def read_instances(instance_file, language='english'):
    with open(instance_file, encoding='utf-8') as file:
        instances = {}
        count = 0
        max_length = 0
        for sentence in file:
            if language == 'english':
                splits = sentence.split()
                key = splits[0]
                words = splits[1:]
                max_length = max(max_length, len(words))
            elif language == 'chinese':
                print('[ERROR] handler for chinese not finish yet!')
                exit(1)
            else:
                print('[ERROR] unsupported language!')
                exit(1)
            instances[key] = words
            count += 1
    print('[INFO] get {} instance sentence, max length is {} words.'.format(count, max_length))
    return instances


#build the vocab (will ignore words that don't always appearance)
def build_vocab(instances, min_word_count = 0):
    vocab = set(word for key in instances for word in instances[key])

    word2idx = {
        constants.BOS_WORD: constants.BOS,
        constants.EOS_WORD: constants.EOS,
        constants.PAD_WORD: constants.PAD,
        constants.UNK_WORD: constants.UNK
    }

    word_count = {word:0 for word in vocab}

    for key in instances:
        for word in instances[key]:
                word_count[word] += 1

    ignored = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored += 1

    print('[INFO] get vocab of size {} (with conrol words).'.format(len(word2idx)))
    if min_word_count > 0:
        print('[INFO] trimmed by min word count {}, {} words is ignored.'.format(min_word_count, ignored))
    return word2idx


#save the vocab in pytorch format
def save_vocab(vocab, vocab_file):
    torch.save(vocab, vocab_file)
    print('[INFO] vocab_file is saved to {}.'.format(vocab_file))


#replace the words by index readed from vocab file, or reconstruct the word index by word
def apply_vocab(instances, vocab_file, mode):
    word2idx = torch.load(vocab_file)

    applied = {}
    if mode == 'word2idx':
        for key in instances:
            applied[key] = np.array([word2idx[word] if word in word2idx else constants.UNK for word in instances[key]])
    elif mode == 'idx2word':
        idx2word = {index:word for word, index in word2idx.items()}
        for key in instances:
            applied[key] = [idx2word[index] if index in idx2word else constants.UNK_WORD for index in instances[key]]
    else:
        print('[ERROR] invaldi mode string.')
        exit(1)

    print('[INFO] vocab with {} words is applied to label, vocab file is {}.'.format(len(word2idx), vocab_file))
    return applied


#add BOS and EOS to the instances index
def add_control_words_index(instances_index):
    for key in instances_index:
        #add two control words
        instances_index[key] = np.array([constants.BOS] + instances_index[key] + [constants.EOS])
    return instances_index


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