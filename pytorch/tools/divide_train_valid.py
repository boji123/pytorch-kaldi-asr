''' 用于将数据集分割成train和valid集合 '''
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', required=True)
    parser.add_argument('-tgt', required=True)
    parser.add_argument('-valid_rate', type=float, default=0.1)
    parser.add_argument('-output_dir', required=True)
    
    opt = parser.parse_args()
    
    src_sents = []
    with open(opt.src, encoding='utf-8') as f:
        for sent in f:
            src_sents += [sent]

    tgt_sents = []
    with open(opt.tgt, encoding='utf-8') as f:
        for sent in f:
            tgt_sents += [sent]

    paired_insts = list(zip(src_sents, tgt_sents))
    print('get {} instance pair'.format(len(paired_insts)))

    valid_size = int(len(paired_insts) * opt.valid_rate)
    print('{}% instances will keep for validation ({} instances)'.format(opt.valid_rate*100,valid_size))

    random.shuffle(paired_insts)

    train_pair = paired_insts[valid_size:]
    train_src, train_tgt = zip(*train_pair)
    valid_pair = paired_insts[:valid_size]
    valid_src, valid_tgt = zip(*valid_pair)

    print('instances is shuffled and divided')

    def write_iterable(iterable, filename, encoding='utf-8'):
        with open(filename, 'wb') as f:
            for item in iterable:
                f.write(item.encode(encoding))

    write_iterable(train_src, opt.output_dir + '/train_src.txt')
    write_iterable(train_tgt, opt.output_dir + '/train_tgt.txt')
    write_iterable(valid_src, opt.output_dir + '/valid_src.txt')
    write_iterable(valid_tgt, opt.output_dir + '/valid_tgt.txt')

    print('4 new file created')

if __name__ == '__main__':
    main()