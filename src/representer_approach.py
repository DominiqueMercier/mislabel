
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class softmax(nn.Module):
    """Class to compute softmax

    Arguments:
        nn {obj} -- neural net class
    """

    def __init__(self, W):
        """Init the variable

        Arguments:
            W {array} -- weight variable
        """
        super(softmax, self).__init__()
        dtype = torch.cuda.FloatTensor
        self.W = Variable(torch.from_numpy(W).type(dtype), requires_grad=True)

    def forward(self, x, y):
        """Forward operation to calculate softmax

        Arguments:
            x {array} -- data array
            y {array} -- label array

        Returns:
            tuple -- phi and l2 for the regularizer
        """
        # calculate loss for the loss function and L2 regularizer
        D = (torch.matmul(x, self.W))
        D_max, _ = torch.max(D, dim=1, keepdim=True)
        D = D-D_max
        A = torch.log(torch.sum(torch.exp(D), dim=1))
        B = torch.sum(D*y, dim=1)
        Phi = torch.sum(A-B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi, L2)


def softmax_np(x):
    """Computes softmax

    Arguments:
        x {array} -- data array

    Returns:
        array -- softmax avlues
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def load_data(options):
    """Loads the data for the representer computation

    Arguments:
        options {dict} -- parameter dictionary

    Returns:
        tuple -- feautre set, outputs and model
    """
    train_feature = np.load(options.representer_path + 'feature_dataset.npy', allow_pickle=True)
    train_output = np.load(options.representer_path + 'label_dataset.npy', allow_pickle=True)
    params = np.load(options.representer_path + 'model_last_weights.npy', allow_pickle=True)
    weight = params[0]
    bias = params[1]
    weight = np.transpose(weight, [1, 0])
    weight = np.transpose(np.concatenate(
        [weight, np.expand_dims(bias, 1)], axis=1))
    train_feature = np.concatenate(
        [train_feature, np.ones((train_feature.shape[0], 1))], axis=1)
    #train_output = softmax_np(train_output)
    model = softmax(weight)
    model.cuda()
    return (train_feature, train_output, model)


def to_np(x):
    """Transforms a gpu object ot a cou object

    Arguments:
        x {array} -- gpu data

    Returns:
        array -- cpu data
    """
    return x.data.cpu().numpy()


def backtracking_line_search(optimizer, model, grad, x, y, val, beta, N, lmbd):
    """implmentation for backtracking line search

    Arguments:
        optimizer {obj} -- optimizer
        model {model} -- model
        grad {array} -- gradient array
        x {array} -- data array
        y {array} -- label array
        val {float} -- value
        beta {float} -- beta
        N {float} -- N
        lmbd {float} -- lambda
    """
    t = 10.0
    beta = 0.5
    W_O = to_np(model.W)
    grad_np = to_np(grad)
    while(True):
        dtype = torch.cuda.FloatTensor
        model.W = Variable(torch.from_numpy(
            W_O-t*grad_np).type(dtype), requires_grad=True)
        val_n = 0.0
        (Phi, L2) = model(x, y)
        val_n = Phi/N + L2*lmbd
        if t < 0.0000000001:
            print("t too small")
            break
        if to_np(val_n - val + t*torch.norm(grad)**2/2) >= 0:
            t = beta * t
        else:
            break


def softmax_torch(temp, N):
    """calculation for softmax in torch, which avoids numerical overflow

    Arguments:
        temp {array} -- temp
        N {float} -- N

    Returns:
        array -- softmax values
    """
    max_value, _ = torch.max(temp, 1, keepdim=True)
    temp = temp-max_value
    D_exp = torch.exp(temp)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N, 1)
    return D_exp.div(D_exp_sum.expand_as(D_exp))


def train(X, Y, model, epochs, lmbd):
    """Comptes the representer values

    Arguments:
        X {array} -- data array
        Y {array} -- label array
        model {model} -- inference model
        epochs {int} -- number of epochs to train
        lmbd {flaot} -- lambda

    Returns:
        [type] -- [description]
    """
    x = Variable(torch.FloatTensor(X).cuda())
    y = Variable(torch.FloatTensor(Y).cuda())
    N = len(Y)
    min_loss = 10000.0
    optimizer = optim.SGD([model.W], lr=1.0)
    for epoch in range(epochs):
        sum_loss = 0
        phi_loss = 0
        optimizer.zero_grad()
        (Phi, L2) = model(x, y)
        loss = L2*lmbd + Phi/N
        phi_loss += to_np(Phi/N)
        loss.backward()
        temp_W = model.W.data
        grad_loss = to_np(torch.mean(torch.abs(model.W.grad)))
        # save the W with lowest loss
        if grad_loss < min_loss:
            if epoch == 0:
                init_grad = grad_loss
            min_loss = grad_loss
            best_W = temp_W
            if min_loss < init_grad/200:
                print('stopping criteria reached in epoch :{}'.format(epoch))
                break
        backtracking_line_search(
            optimizer, model, model.W.grad, x, y, loss, 0.5, N, lmbd)
        if epoch % 100 == 0:
            print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(
                epoch, to_np(loss), phi_loss, grad_loss))

    # caluculate w based on the representer theorem's decomposition
    temp = torch.matmul(x, Variable(best_W))
    softmax_value = softmax_torch(temp, N)
    # derivative of softmax cross entropy
    weight_matrix = softmax_value-y
    weight_matrix = torch.div(weight_matrix, (-2.0*lmbd*N))

    w = torch.matmul(torch.t(x), weight_matrix)

    # calculate y_p, which is the prediction based on decomposition of w by representer theorem
    temp = torch.matmul(x, w.cuda())

    softmax_value = softmax_torch(temp, N)
    y_p = to_np(softmax_value)

    print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
    print(np.mean(np.abs(to_np(y)-y_p)))

    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
    y = to_np(y)
    corr, _ = (pearsonr(y.flatten(), (y_p).flatten()))
    print(corr)
    sys.stdout.flush()
    return to_np(weight_matrix)


def perform_representer_influence(options):
    """Computees and saves the representer values

    Arguments:
        options {dict} -- parameter dictionary
    """
    print('Compute Representer Influence')
    x, y, model = load_data(options)
    weight_matrix = train(x, y, model, 1000, 0.003)
    np.save(options.representer_path +
            'representer_influence.npy', [weight_matrix, y])
