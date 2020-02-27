from mlxtend.data import loadlocal_mnist
import numpy as np

def F(scores):
    return np.exp(scores) / (1 + np.exp(scores))



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
            predictions = F(scores)

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
            if (step + 1) % 1 == 0:
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
        predictions = F(scores)
        return predictions


def eval_pred(pred, Y):
    hit = 0
    for idx, y_hat in enumerate(pred):
        if y_hat == Y[idx]:
            hit += 1
    return hit / len(pred)


mnist = '../data/mnist'

X, y = loadlocal_mnist(
        images_path= mnist + '/train-images-idx3-ubyte',
        labels_path= mnist + '/train-labels-idx1-ubyte')

# Make it binary classification
new_X = []
new_y = []
for i in range(len(X)):
    if y[i] == 0 or y[i] == 1:
        new_X.append(X[i])
        new_y.append(y[i])
X, y = np.array(new_X), np.array(new_y)

# Normalize
X = X.astype(float) / 255

# Split the data
tr_X, tr_y = X[:10000], y[:10000]
dev_X, dev_y = X[10000:], y[10000:]



model = LogisticRegressionModel(tr_X.shape[1])
model.train(tr_X, tr_y,
                 num_steps=20, learning_rate=5e-5, dev_feat=dev_X, dev_tgt=dev_y)
pred = np.ceil( model.prediction(tr_X) - 0.5)
print('Train Acc. {:.4f}'.format(eval_pred(pred, tr_y)))
pred = np.ceil( model.prediction(dev_X) - 0.5)
print('Dev. Acc. {:.4f}'.format(eval_pred(pred, dev_y)))
