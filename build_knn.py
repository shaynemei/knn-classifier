import numpy as np
import sys
from collections import Counter
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


def load_svm(path, vocab, use_pipe=False):
    with open(path) as f:
        num_entries = 0
        for line in f.readlines():
            num_entries += 1
            if not use_pipe:
                line = line.strip().split()
                vocab |= {feature.split(':')[0] for feature in line[1:]}
    vocab = list(vocab)
    vocab.sort()
    
    with open(path) as f:
        data = np.zeros((num_entries, len(vocab) + 1))
        for i, line in enumerate(f.readlines()):
            line = line.strip().split()
            y = y_dict[line[0]]
            data[i][len(vocab)] = y_dict[line[0]]
            features = {feature.split(':')[0]:int(feature.split(':')[1]) for feature in line[1:]}
            for j, word in enumerate(vocab):
                if word in features:
                    data[i][j] = features[word]
    return data, vocab

#def eucli_dist(vec1, vec2):
    # Euclidean distance is l2 norm and the default value of ord parameter in np.linalg.norm is 2.
    #return np.linalg.norm(vec1 - vec2)

#def cosine_dist(vec1, vec2):
    #return 1 - ((vec1.dot(vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class kNN:
    def __init__(self):
        self.sim_func = {1: "euclidean",
                        2: "cosine"}
        self.train = None
    
    def fit_predict(self, train, test, k_VAL, sim_func_id, return_probs=False):
        # create one distance matrix for all (improves speed)
        self.train = train
        len_train = len(train)
        data = np.concatenate((train[:, :-1], test[:, :-1])) 
        matrix = squareform(pdist(data, metric=self.sim_func[sim_func_id])) 
        dist_train =  matrix[:len_train, :len_train-1]
        dist_test = matrix[len_train:, :len_train-1]
        
        if return_probs:
            res_train = [self.get_prob(query, k_VAL) for query in dist_train]
            res_test = [self.get_prob(query, k_VAL) for query in dist_test]
        else:
            res_train = [stats.mode(list(map(get_cls, np.argsort(query)[:k_VAL])))[0][0] for query in dist_train]
            res_test = [stats.mode(list(map(get_cls, np.argsort(query)[:k_VAL])))[0][0] for query in dist_test]
        
        return res_train, res_test
    
    def get_cls(self, index):
        return self.train[index, -1]
    
    def get_prob(self, query, k_VAL):
        labels_counts = Counter(sorted(list(map(self.get_cls, np.argsort(query)[:k_VAL]))))
        prob0 = labels_counts[0] / k_VAL
        prob1 = labels_counts[1] / k_VAL
        prob2 = labels_counts[2] / k_VAL
        return sorted([(prob0, 0), (prob1, 1), (prob2, 2)], reverse=True)


def print_confusion_matrix(res, truth, labels):
    labels = y_dict.keys()
    counts_dict = count_res(res, truth)
    print("                  ", end="")
    for label in labels:
        print(f"{label} ", end="")
    print()
    for i, label in enumerate(labels):
        col0 = counts_dict[i] if i in counts_dict else 0
        col1 = counts_dict[i + 3] if i+3 in counts_dict else 0
        col2 = counts_dict[i + 6] if i+6 in counts_dict else 0
        print(f"{label}\t{col0}\t\t{col1}\t\t{col2}")
        
def print_accuracy(res, truth):
    counts_dict = count_res(res, truth)
    col0 = counts_dict[0] if 0 in counts_dict else 0
    col1 = counts_dict[4] if 4 in counts_dict else 0
    col2 = counts_dict[8] if 8 in counts_dict else 0
    print((col0+col1+col2)/len(truth))

def count_res(res, truth):
    unique, counts = np.unique(np.sum([res, truth*3], axis=0), return_counts=True) 
    counts_dict = dict()
    for i, j in zip(unique, counts):
        counts_dict[i] = j
    return counts_dict

#Usage: build kNN.sh training_data test_data k_val similarity_func sys_output > acc_file
if __name__ == "__main__":
    PATH_TRAIN = sys.argv[1]
    PATH_TEST = sys.argv[2]
    k_VAL = int(sys.argv[3])
    SIM_FUNC_ID = int(sys.argv[4])
    out_sys = sys.argv[5]

    y_dict = dict({"talk.politics.guns": 0,
             "talk.politics.mideast": 1,
             "talk.politics.misc": 2})
    labels = list(y_dict.keys())
    vocab = set() 
    train, vocab = load_svm(PATH_TRAIN, vocab)
    test, vocab = load_svm(PATH_TEST, vocab, True)

    knn = kNN()
    res_train_probs, res_test_probs = knn.fit_predict(train, test, k_VAL, SIM_FUNC_ID, return_probs=True)
    res_train = [probs[0][1] for probs in res_train_probs]
    res_test = [probs[0][1] for probs in res_test_probs]
    truth_train = train[:, -1]
    truth_test = test[:, -1]


    with open(out_sys, 'w') as f:
        f.write("%%%%% training data:\n")
        for i, probs in enumerate(res_train_probs):
            f.write(f"array:{i} {labels[int(train[i, -1])]} {labels[int(probs[0][1])]} {probs[0][0]} {labels[int(probs[1][1])]} {probs[1][0]} {labels[int(probs[2][1])]} {probs[2][0]}\n")

        f.write("%%%%% test data:\n")
        for i, probs in enumerate(res_test_probs):
            f.write(f"array:{i} {labels[int(test[i, -1])]} {labels[int(probs[0][1])]} {probs[0][0]} {labels[int(probs[1][1])]} {probs[1][0]} {labels[int(probs[2][1])]} {probs[2][0]}\n")

    print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_train, truth_train, labels)
    print()
    print("Training accuracy: ", end="")
    print_accuracy(res_train, truth_train)
    print("\n")
    print("Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_test, truth_test, labels)
    print()
    print("Test accuracy: ", end="")
    print_accuracy(res_test, truth_test)