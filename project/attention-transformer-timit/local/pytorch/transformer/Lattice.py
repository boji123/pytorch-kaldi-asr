#author: boji
#create an edge list to save the decode result with beam

import numpy as np
from utils import constants

class Lattice(object):
    def __init__(self, max_length, beam_size):
        self.max_length =  max_length
        self.curr_length = 0
        self.beam_size = beam_size
        self.edges = [[-1, constants.BOS, 0]] #list of [prev_edge, word_state_id, sum_weight]
        self.curr_edge_index = [0] #edge that reach </s> should be considered
        self.num_curr_active = 1      
        self.done = False

    #returns the index of active edge
    def get_active_edge(self, edge_index):
        indexs = [index for index in edge_index if self.edges[index][1] != constants.EOS]
        return indexs 

    def get_end_edge(self, edge_index):
        indexs = [index for index in edge_index if self.edges[index][1] == constants.EOS]
        return indexs 

    #input edge indexs, return weights
    def get_weights_by_indexs(self, edge_indexs):
        weights = [self.edges[index][2] for index in edge_indexs]
        return weights

    def get_words_by_indexs(self, edge_indexs):
        words = [self.edges[index][1] for index in edge_indexs]
        return words

    def advance(self, weights):
        active_edge_index = self.get_active_edge(self.curr_edge_index)
        if len(active_edge_index) == 0:
            print('[WARNING] decode already finish!')
            return True
        num_words = weights.shape[1]
        if len(self.edges) == 1:
            sum_weights = weights[0]
        else:
            active_weights = self.get_weights_by_indexs(active_edge_index)
            sum_weights = weights.flatten() + np.array(active_weights).repeat(num_words)

        prev_edge_index = np.array(active_edge_index).repeat(num_words)
        num_active = len(prev_edge_index)

        end_edge_index = self.get_end_edge(self.curr_edge_index)
        end_weights = self.get_weights_by_indexs(end_edge_index)

        # append all of the probable weights and keep the best n.
        sum_weights = np.append(sum_weights, end_weights)
        # minues can inverse the order, from largest to smallest
        best_word_index = np.argsort(-sum_weights)[:self.beam_size]

        #add arc to list
        curr_edge_index = []
        #use following code to ensure the weight order
        for index in best_word_index:
            if index < num_active:
                prev_edge = prev_edge_index[index]
                word = index % num_words
                weight = sum_weights[index]
                edge = [prev_edge, word, weight]
                edge_index = len(self.edges)
                self.edges += [edge]
                curr_edge_index += [edge_index]
            else:
                edge_index = end_edge_index[index - num_active]
                curr_edge_index += [edge_index]
        #update curr edge list
        self.curr_edge_index = curr_edge_index
        self.curr_length += 1

        self.num_curr_active = len(self.get_active_edge(self.curr_edge_index))
        if self.num_curr_active == 0 or self.curr_length > self.max_length:
            self.done = True #decode finish

        return self.done


    #given stop index, it will start from index reading forward to output a sequence
    def get_sequence(self, index):
        result = []
        while(index > -1):
            result += [self.edges[index][1]]
            index = self.edges[index][0]
        #return a inverse result
        return result[::-1]

    def get_results(self, mode='all'):
        if mode == 'all': #return all edge in beam
            result_indexs = self.curr_edge_index
        elif mode == 'active': #return edge that not reach to end yer
            result_indexs = self.get_active_edge(self.curr_edge_index)
        elif mode == 'end': #return edge that reach to end
            result_indexs = self.get_end_edge(self.curr_edge_index)

        results = []
        for index in result_indexs:
            results += [self.get_sequence(index)]

        weights = self.get_weights_by_indexs(result_indexs)

        return results, weights

def main():
    max_length = 10
    beam_size = 3
    lattice = Lattice(max_length,beam_size)
    weight_arr = [[-99, -99, -99, -4, -3, -2, -1],
                  [-99, -99, -99, -4, -3, -2, -1],
                  [-99, -99, -99, -4, -3, -2, -1]]
    lattice.advance(np.array(weight_arr))

    weight_arr = [[-99, -99, -99, -1.5, -2, -3, -4],
                  [-99, -99, -99, -1.5, -3, -4, -2],
                  [-99, -99, -99, -1.5, -4, -3, -2]]
    lattice.advance(np.array(weight_arr))

    weight_arr = [[-99, -99, -99, -1.5, -2, -3, -4]]
    lattice.advance(np.array(weight_arr))

    if lattice.done:
        results, weights = lattice.get_results()
        print(results)
        print(weights)
        print(lattice.edges)

if __name__ == '__main__':
    main()