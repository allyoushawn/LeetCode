#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
import pickle


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))



class LogisticRegressionModel:

    def log_likelihood(features, target, weights):
        scores = np.dot(features, weights)
        ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
        return ll

    def __init__(self, feat_dim):
        self.weights = np.random.rand((feat_dim))



    def train(self, features, target, num_steps, learning_rate, verbose=False, dev_feat=None, dev_tgt=None):
        # Ref: https://beckernick.github.io/logistic-regression-from-scratch/



        lr = learning_rate

        best_w = None
        best_dev = -1
        prev_grad = None
        for step in range(num_steps):
            scores = np.dot(features, self.weights)
            predictions = sigmoid(scores)

            # Update weights with gradient
            output_error_signal = target - predictions
            gradient = np.dot(features.T, output_error_signal)

            # Momentum
            if prev_grad is not None: gradient += 0.9 * prev_grad
            self.weights += learning_rate * gradient
            prev_grad = gradient

            # Halving lr
            if (step + 1) % 500 == 0:
                lr /= 2

            # Print log-likelihood every so often
            if (step + 1) % 50 == 0:
                if dev_feat is not None and dev_tgt is not None:
                    pred = self.prediction(dev_feat)
                    pred = np.ceil(pred - 0.5)
                    dev_score = eval_pred(pred, dev_tgt)
                    print('Step {}, dev. Score: {:.4f}'.format(step + 1, dev_score))
                    if dev_score > best_dev:
                        best_w = self.weights.copy()
                        best_dev = dev_score
                    else:
                        self.weights = best_w
                        lr /= 2
                        if step > 2500:
                            break

    def prediction(self, features):
        scores = np.dot(features, self.weights)
        predictions = sigmoid(scores)
        return predictions


class PLAModel:
    def __init__(self, feat_dim):
        self.weights = np.zeros(feat_dim)
        self.weight_ensemble = []

    def train(self, features, target, num_batch, verbose=False, dev_feat=None, dev_tgt=None):
        target = 2 * (target - 0.5) # Make target from [1,0] to [1, -1]
        best_w = None
        best_dev = -1
        for batch in range(num_batch):
            for i in range(len(features)):
                if np.dot(self.weights, features[i]) * target[i] <= 0:
                    self.weights += target[i] * features[i]

            if dev_feat is not None and dev_tgt is not None:
                pred = self.prediction(dev_feat)
                dev_score = eval_pred(pred, dev_tgt)
                print('Batch {}, dev. Score: {:.4f}'.format(batch + 1, dev_score))
                if dev_score > best_dev or batch < 50:
                    best_w = self.weights.copy()
                    self.weight_ensemble.append(self.weights.copy())
                    best_dev = dev_score
                else:
                    self.weights = np.array(self.weight_ensemble).sum(axis=0) / len(self.weight_ensemble)
                    break


    def prediction(self, features):
        scores = np.dot(features, self.weights)
        predictions = (np.sign(scores) + 1) / 2
        return predictions


def prep_data(filename='all_names.csv'):
    tr = []
    dev = []
    tst = []
    count = 0
    with open(filename) as f:
        for line_id, line in enumerate(f.readlines()):
            if line_id == 0: continue
            elements = line.rstrip().split(',')
            # Preprocessing the name
            name = elements[1].lower().split()[0]
            if elements[3] == 'Test':
                tst.append((name, elements[2]))
            else:
                count += 1
                if count % 9 == 0:
                    dev.append((name, elements[2]))
                else:
                    tr.append((name, elements[2]))
    return tr, dev, tst

def build_voc(tr):
    voc = defaultdict(lambda: len(voc))
    UNK = voc['UNK']
    for i, data in enumerate(tr):
        name, gender = data
        # Unigram
        for i, c in enumerate(name):
            voc[c]
            voc['order-'+c]
        # Bigram
        for i, c in enumerate(name):
            if i == len(name) - 1:
                break
            voc[name[i:i+2]]
        # Trigram
        for i, c in enumerate(name):
            if i == len(name) - 2:
                break
            voc[name[i:i+3]]
        if len(name) >= 1:
            voc[ 'last-letter=' + name[-1]]
        if len(name) >= 2:
            voc[ 'last-two-letters=' + name[-2:]]
        if len(name) >= 3:
            voc[ 'last-three-letters=' + name[-3:]]
    voc = defaultdict(lambda: UNK, voc)
    voc_dim = len(voc)
    return voc, voc_dim


def vectorize(split, voc, voc_dim):
    X = []
    Y = []
    for data in split:
        vec = [0] * voc_dim
        name = data[0]
        # Unigram
        for i, c in enumerate(name):
            vec[voc[c]] += 1
            vec[voc['order-'+c]] += i
        # Bigram
        for i in range(len(name)):
            if i == len(name) - 1: break
            vec[voc[name[i:i+2]]] += 1
        # Trigram
        for i in range(len(name)):
            if i == len(name) - 2: break
            vec[voc[name[i:i+3]]] += 1
        if len(name) >= 1:
            vec[voc[ 'last-letter=' + name[-1]]] += 1
        if len(name) >= 2:
            vec[voc[ 'last-two-letters=' + name[-2:]]] += 1
        if len(name) >= 3:
            vec[voc[ 'last-three-letters=' + name[-3:]]] += 1
        X.append(vec)
        if data[1] == 'Male':
            Y.append(0)
        else:
            Y.append(1)
    return np.array(X), np.array(Y)


def eval_pred(pred, Y):
    hit = 0
    for idx, y_hat in enumerate(pred):
        if y_hat == Y[idx]:
            hit += 1
    return hit / len(pred)


if __name__ == '__main__':
    # Feature extraction ref:
    # https://blog.ayoungprogrammer.com/2016/04/determining-gender-of-name-with-80.html/
    tr, dev, tst = prep_data('all_names.csv')
    print('Building vocabulary...')
    voc, voc_dim = build_voc(tr)
    print('Vectorizing data...')
    tr_X, tr_Y = vectorize(tr, voc, voc_dim)
    dev_X, dev_Y = vectorize(dev, voc, voc_dim)
    tst_X, tst_Y = vectorize(tst, voc, voc_dim)

    # PCA to speed up training
    print('Performing PCA...')
    pca = PCA(n_components=3000)
    pca.fit(tr_X)
    print('Applying PCA...')
    tr_X = pca.transform(tr_X)
    dev_X = pca.transform(dev_X)
    tst_X = pca.transform(tst_X)

    # LR Model
    # 0.94511 on test set.
    model = LogisticRegressionModel(tr_X.shape[1])
    model.train(tr_X, tr_Y,
                     num_steps=10000, learning_rate=5e-5, dev_feat=dev_X, dev_tgt=dev_Y)
    pred = np.ceil( model.prediction(tr_X) - 0.5)
    print('Train Acc. {:.4f}'.format(eval_pred(pred, tr_Y)))
    pred = np.ceil( model.prediction(dev_X) - 0.5)
    print('Dev. Acc. {:.4f}'.format(eval_pred(pred, dev_Y)))
    pred = np.ceil( model.prediction(tst_X) - 0.5)
    print('Test. Acc. {:.4f}'.format(eval_pred(pred, tst_Y)))
    quit()

    '''
    # PLA Performance is 0.7993 on test set
    model = PLAModel(tr_X.shape[1])
    model.train(tr_X, tr_Y,
                     num_batch=1000, dev_feat=dev_X, dev_tgt=dev_Y)
    '''

