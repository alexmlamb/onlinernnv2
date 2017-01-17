#!/usr/bin/env python
import gzip
import cPickle as pickle
import theano
import theano.tensor as T
import numpy.random as rng
import random
import lasagne
import os

slurm_name = os.environ["SLURM_JOB_ID"]

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid

trainy = trainy.astype('int32')
validy = validy.astype('int32')

import sys
sys.path.append("/u/lambalex/DeepLearning/lambwalker")
sys.path.append("/u/lambalex/DeepLearning/lambwalker/lib")

from viz import plot_images

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

def join(a,b):
    return T.concatenate([a,b],axis=1)

'''

'''
def clip_updates(updates):
    for param in updates:
        updates[param] = T.clip(updates[param], -0.1, 0.1)
    return updates

def init_params():
    params = {}

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(784*2,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(1,)).astype('float32'))

    return params

def bn(inp):
    return (inp - inp.mean(axis=1,keepdims=True)) / (0.001 + inp.std(axis=1,keepdims=True))

def d_net(p,x1,x2):

    x = join(x1,x2)

    h1 = T.nnet.relu(T.dot(x, p['w1']) + p['b1'],alpha=0.02)
    h2 = T.nnet.relu(T.dot(h1,p['w2']) + p['b2'],alpha=0.02)
    h3 = T.dot(h2,p['w3']) + p['b3']

    y = h3

    return y

'''
    This pass goes from real x to noise.  
'''
def forward(params, x):

    xn = x*1.0

    #xn = xn - 1.0 * T.grad(d_net(params,x,xn).mean(), xn)

    xn = T.clip(xn + 0.1 * srng.normal(size = xn.shape).astype('float32'),0.0,1.0)

    return xn

def backward(params, x):
    
    xl = x*1.0

    xl = xl + 0.01 * T.grad(d_net(params,xl,x).mean(), xl)

    return xl

if __name__ == "__main__":
    pass

    #runs forward or backward step.  
    #forward_method = theano.function()
    #backward_method = theano.function()

    x = T.matrix()
    params = init_params()

    x1 = T.matrix()
    x2 = T.matrix()
    score = d_net(params, x1, x2)

    compute_score = theano.function([x1,x2], score)

    xn_forward = forward(params,x)
    xl_backward = backward(params,x)

    #Mark as 1.0
    forward_loss = -1.0 * d_net(params, x, xn_forward).mean()
    forward_loss += d_net(params, xn_forward, x).mean()
    forward_loss += T.abs_(d_net(params,x,x)).mean()

    #Mark as negative.  
    backward_loss = 0.0 * d_net(params, xl_backward, x).mean()

    forward_updates = clip_updates(lasagne.updates.adam(forward_loss, params.values()))
    backward_updates = clip_updates(lasagne.updates.adam(backward_loss, params.values()))

    #need to add parameter clipping.  

    forward_method_train = theano.function([x], outputs = xn_forward, updates = forward_updates)
    backward_method_train = theano.function([x], outputs = xl_backward, updates = backward_updates)

    backward_method = theano.function([x], outputs = xl_backward)

    print "derp"

    for iteration in range(0,5000):
        r = random.randint(0,40000)
        xb = trainx[r:r+64]

        xq = 0.1 * rng.normal(size = (64,784)).astype('float32')

        if iteration % 100 == 0:
            print "iteration", iteration

        for step in range(0,10):
            xq = backward_method_train(xq)
            xb = forward_method_train(xb)

            if iteration % 100 == 0:
                plot_images(xb.reshape(64,1,28,28), "plots/real/" + str(step), title=slurm_name+"_"+str(iteration))

        if iteration % 100 == 0:

            for step in range(0,30):
                xb_next = backward_method(xb)

                print step, compute_score(xb, xb_next)

                xb = xb_next

                plot_images(xb.reshape(64,1,28,28), "plots/rec/" + str(step), title=slurm_name+"_"+str(iteration))



