# a test file script, leaved by the developer
# maybe helpful when building your own project.
import argparse
from utils import instances_handler
import sys


def test_vocab():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_instances_file', required=True)
    parser.add_argument('-save_vocab_file', required=True)
    parser.add_argument('-min_word_count', type=int, default=0)
    opt = parser.parse_args()

    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] preparing vocabulary for output label')
    instances = instances_handler.read_instances(opt.read_instances_file)
    vocab = instances_handler.build_vocab(instances)
    instances_handler.save_vocab(vocab, opt.save_vocab_file)

    print(instances['sw02053-B_020556-020891'])
    instances_index = instances_handler.apply_vocab(instances, opt.save_vocab_file, 'word2idx')
    print(instances_index['sw02053-B_020556-020891'])
    instances = instances_handler.apply_vocab(instances_index, opt.save_vocab_file, 'idx2word')
    print(instances['sw02053-B_020556-020891'])


#simulating the parameter passing
def set_arg(arg_list):
    sys.argv = [sys.argv[0]]
    for arg in arg_list:
        sys.argv.append(arg)

if __name__ == '__main__':
    set_arg("-read_instances_file data/text -save_vocab_file exp/vocab.torch".split())
    test_vocab()
