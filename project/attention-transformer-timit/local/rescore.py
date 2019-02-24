#author : baiji
#reweight the sentence output with different lm weight and output the best result
#input format:
#a. decoding result (table of key, score, decode result)
#b. language model score (table of score list with the same order as decode result table)
#c. list of different lm scale (will scale the lm score and output result file for each weight)
#d. save dir (will create series file to save result of different weight)

import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-decode_file', required=True)
    parser.add_argument('-lm_score', required=True)
    parser.add_argument('-save_dir', required=True)
    #inv_weight_list means weight number will work as denominator
    # 2-> 1/2
    parser.add_argument('-inv_weight_list', required=True)
    opt = parser.parse_args()

    #example: '5,10,15,20,25,30,35,40'
    #in current end2end model, model score for best result are around -1 ~ -30
    #while language model score are around -20 ~ -50 
    weights = opt.inv_weight_list.split(',')
    weights = [float(weight) for weight in weights]

    print('[PROCEDURE] start rescoring...')
    score_list = {}
    with open(opt.decode_file, encoding='utf-8') as decode_file, \
      open(opt.lm_score, encoding='utf-8') as lm_scores:
        for de in decode_file:
            (key, de_score, result) = de.split('\t')
            lm_score = lm_scores.readline()
            de_score = float(de_score.strip())
            lm_score = float(lm_score.strip())
            result = result.strip()
            if not score_list.__contains__(key):
                score_list[key] = [[de_score],[lm_score],[result]]
            else:
                score_list[key][0] += [de_score]
                score_list[key][1] += [lm_score]
                score_list[key][2] += [result]

    for key in score_list:
        score_list[key][0] = np.array(score_list[key][0])
        score_list[key][1] = np.array(score_list[key][1])

    print('[INFO] required file loaded.')

    for weight in weights:
        print('[INFO] handling inv weight {}'.format(weight))
        #one inv lm weigth for one file
        file_name = opt.save_dir + '/rescore_' + str(weight)
        with open(file_name, 'w', encoding='utf-8') as file:
            for key in score_list:
                rescore_list = score_list[key]
                #0 as model score, 1 as lm score, 2 as result
                scores = rescore_list[0] + rescore_list[1]/weight
                max_id = scores.argmax()
                result = rescore_list[2][max_id]

                file.write(key + ' ' + result + '\n')
    print('[INFO] rescoring finished')

if __name__ == '__main__':
    main()