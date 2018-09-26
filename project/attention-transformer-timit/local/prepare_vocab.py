#handle and save the vocab
import argparse
from utils import instances_handler


def main():
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


if __name__ == '__main__':
    main()