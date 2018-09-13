''' cut the file encoding utf-8 witn no BOM '''
import argparse
import jieba

parser = argparse.ArgumentParser(description='jieba_cutter.py')

parser.add_argument('-sents_input', required=True,
                    help='sentences to be cut')
parser.add_argument('-cutted_output', required=True,
                    help='cutted sentences output')

opt = parser.parse_args()
print(opt)

count = 0
with open(opt.sents_input, encoding='utf-8') as in_f, open(opt.cutted_output, 'wb') as out_f:
    for sent in in_f:
        count = count + 1
        cuts = jieba.cut(sent)
        
        #必须要调用join把cut的结果读取出来，否则直接打印cuts只是个对象
        sent = " ".join(cuts).split()#用以去掉句末字符\n
        sent = " ".join(sent) + '\n'
        #print(sent)
        out_f.write(sent.encode('utf-8'))

print('[Info] Cut {} sentences from {}'.format(count, opt.sents_input))
print('[Info] Result is saved to {}'.format(opt.cutted_output))
