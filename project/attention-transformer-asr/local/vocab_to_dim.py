#handle and save the vocab
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_vocab_file', required=True)
    opt = parser.parse_args()

    word2idx = torch.load(vocab_file)
    print(len(word2idx))

if __name__ == '__main__':
    main()