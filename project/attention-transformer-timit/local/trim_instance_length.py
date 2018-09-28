#drop the long instance in case memory overflow
import argparse
import os
from utils import instances_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-max_len', type=int, required=True)
    opt = parser.parse_args()

    read_path = {}
    read_path['feats'] = opt.data_dir + '/feats.scp'
    read_path['text'] = opt.data_dir + '/text'
    read_path['length'] = opt.data_dir + '/feats.length'

    for key in read_path:
        if not os.path.exists(read_path[key]):
            print('[ERROR] {} not founded.'.format(read_path[key]))
            exit(1)

    length_dict = {}
    with open(read_path['length'], encoding='utf-8') as file:
        for line in file:
            key, length = line.split()
            length = int(length)
            length_dict[key] = length

    write_path = {}
    write_path['feats'] = opt.data_dir + '/feats.filtered.scp'
    write_path['text'] = opt.data_dir + '/text.filtered'

    print('[INFO] filtering instance with max length {}.'.format(opt.max_len))

    filtered = 0
    total = 0
    with open(read_path['feats'], encoding='utf-8') as rfile, open(write_path['feats'], 'wb') as wfile:
        for line in rfile:
            key = line.split()[0]
            if length_dict[key] < opt.max_len:
                wfile.write(line.encode('utf-8'))
                filtered += 1
            total += 1
        print('[INFO] {}/{} filtered feats successfully saved to {}.'.format(filtered, total, write_path['feats']))

    filtered = 0
    total = 0
    with open(read_path['text'], encoding='utf-8') as rfile, open(write_path['text'], 'wb') as wfile:
        for line in rfile:
            key = line.split()[0]
            if length_dict[key] < opt.max_len:
                wfile.write(line.encode('utf-8'))
                filtered += 1
            total += 1
        print('[INFO] {}/{} filtered text successfully saved to {}.'.format(filtered, total, write_path['text']))


if __name__ == '__main__':
    main()