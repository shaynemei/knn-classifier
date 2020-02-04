from collections import Counter
import numpy as np
import sys

def get_chi2(feat, data):
    # fill contingency table of observation
    counter = Counter(np.sum([data[:, feat], data[:, -1]*2], axis=0))
    obs = np.zeros((3,num_cls+1))
    for i in range(0, 2):
        for j in range(0, num_cls):
            obs[i, j] = counter[j*2 + i]
    for i in range(0, 2):
        obs[i, -1] = np.sum(obs[i,:])
    for j in range(0, num_cls):
        obs[-1, j] = np.sum(obs[:,j])
    obs[-1,-1] = np.sum(obs[-1,:])
    
    # fill contingency table of expectation
    exp = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, num_cls):
            exp[i, j] = (obs[-1,j] * obs[i,-1]) / obs[-1,-1]

    return np.sum(np.square(np.diff(np.stack((np.matrix.flatten(obs[:-1,:-1]), np.matrix.flatten(exp))), axis=0)) / np.matrix.flatten(exp))

#usage: cat input_file | rank_feat_by_chi_square.sh > output_file
#out: featName score docFreq
if __name__ == "__main__":
    inp = sys.stdin.readlines()
    y_dict = dict({"talk.politics.guns": 0,
         "talk.politics.mideast": 1,
         "talk.politics.misc": 2})
    labels = list(y_dict.keys())
    num_cls = len(labels)
    vocab = set() 
    num_entries = 0
    for line in inp:
        num_entries += 1
        line = line.strip().split()
        vocab |= {feature.split(':')[0] for feature in line[1:]}
    vocab = list(vocab)
    len_vocab = len(vocab)
    vocab.sort()

    data = np.zeros((num_entries, len_vocab + 1))
    for i, line in enumerate(inp):
        line = line.strip().split()
        y = y_dict[line[0]]
        data[i][len(vocab)] = y_dict[line[0]]
        features = {feature.split(':')[0] for feature in line[1:]}
        for j, word in enumerate(vocab):
            if word in features:
                data[i][j] = 1
    

    chi2_score = [get_chi2(feat, data) for feat in range(0, len_vocab)]
    sorted_idx = np.argsort(chi2_score)[:-1][::-1]
    for i in sorted_idx:
        print(f"{vocab[i]} {chi2_score[i]} {int(np.sum(data[:,i]))}")
