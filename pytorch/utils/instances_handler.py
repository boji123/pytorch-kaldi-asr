#handling the instance text, generete a dict and return the instance

import torch
from utils import constants

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


#replace the words by index readed from vocab file
def apply_vocab(instances, vocab_file):
    word2idx = torch.load(vocab_file)
    for key in instances:
        instances[key] = [word2idx[word] if word in word2idx else constants.UNK for word in instances[key]]
    return instances
