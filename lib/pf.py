#!/usr/bin/env python
import gzip
import cPickle as pickle
import theano
import theano.tensor as T
import numpy.random as rng
import random
import lasagne
import os
import numpy as np

'''

Q network, P network, and D network.  

Data: 
    a, 0, 0, 0, a
    -a, 0, 0, 0, -a

Waking phase: 
    -Sample x.  
    -Maximize p(x[t]|z[t]) through q(z[t] | z[t-1],x[t])
    -Make q a gaussian.  
    -Try to move D(z[t-1],z[t]) to 0.0.  Train D to say that it's 1.0.  

Sleep phase:
    -Sample p(z[t] | z[t-1])
    -Feed to D(z[t-1],z[t]), try to move to 1.0.  Train D to say that it's 0.0.  

Daydream phase (later).  

P network: 
    z -> z

Q network: 
    z,x -> z

D network:
    z,z -> 1

'''

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
sys.path.append("/u/lambalex/DeepLearning/lambwalker/onlinernnv2")
sys.path.append("/u/lambalex/DeepLearning/lambwalker/onlinernnv2/lib")

from viz import plot_images

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

def join(a,b):
    return T.concatenate([a,b],axis=1)

def join3(a,b,c):
    return T.concatenate([a,b,c],axis=1)

'''

'''
def clip_updates(updates,params2clip):
    for param in updates:
        if param in params2clip:
            updates[param] = T.clip(updates[param], -0.05, 0.05)
    return updates

def init_params_q():
    params = {}

    #x,z -> z.  

    nx = 1

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(nx+1024,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))

    return params

def init_params_s():
    params = {}

    #z -> z.

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024+1024,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))

    return params


def init_params_p():
    params = {}

    nout = 1

    #z -> x

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,nout)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(nout,)).astype('float32'))

    return params

def init_params_d():
    params = {}

    #(z,z) -> 1

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024*2 + 1,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(1,)).astype('float32'))

    return params


def init_params_zr():
    params = {}

    #(z,z) -> 1

    params['w1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    params['w3'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))

    params['b1'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b2'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))
    params['b3'] = theano.shared(0.0 * rng.normal(0,1,size=(1024,)).astype('float32'))

    return params



def bn(inp):
    return (inp - inp.mean(axis=0,keepdims=True)) / (0.001 + inp.std(axis=0,keepdims=True))

def ln(inp):
    return (inp - inp.mean(axis=1,keepdims=True)) / (0.001 + inp.std(axis=1,keepdims=True))

def wn(inp,w):
    print "norm axis 1 wn"
    #can do axis=0
    out = (inp) / (0.01 + T.sqrt((w**2).sum(axis=0)))
    #out = out - out.mean(axis=0,keepdims=True)
    return out

def net_p(p,z):

    #z -> x

    inp = z

    h1 = T.nnet.relu(ln(T.dot(inp, p['w1']) + p['b1']),alpha=0.02)
    h1 *= T.cast(srng.binomial(n=1,p=0.5,size=h1.shape),'float32')
    h2 = T.nnet.relu(ln(T.dot(h1,p['w2']) + p['b2']),alpha=0.02)
    h2 *= T.cast(srng.binomial(n=1,p=0.5,size=h2.shape),'float32')
    h3 = T.dot(h2,p['w3']) + p['b3']

    y = h3

    return y

def net_s(p,z,step):

    #z -> z

    #z *= T.cast(srng.binomial(n=1,p=0.1,size=z.shape),'float32')

    z2 = srng.normal(size=(z.shape[0],1024)) + 0.0*T.cast(step,'float32')

    inp = join(z,z2)

    h1 = T.nnet.relu(ln(T.dot(inp, p['w1']) + p['b1']),alpha=0.02)
    h1 *= T.cast(srng.binomial(n=1,p=0.5,size=h1.shape),'float32')
    h2 = T.nnet.relu(ln(T.dot(h1,p['w2']) + p['b2']),alpha=0.02)
    h2 *= T.cast(srng.binomial(n=1,p=0.5,size=h2.shape),'float32')
    h3 = T.dot(h2,p['w3']) + p['b3']

    #zn = srng.normal(size=mu.shape) * sigma + mu
    
    y = h3

    return y

def net_zr(p,z):

    #z -> z

    inp = z

    h1 = T.nnet.relu(ln(T.dot(inp, p['w1']) + p['b1']),alpha=0.02)
    h2 = T.nnet.relu(ln(T.dot(h1,p['w2']) + p['b2']),alpha=0.02)
    h3 = T.dot(h2,p['w3']) + p['b3']

    y = h3

    return y

def net_q(p,x,z):

    #(x,z) -> z

    inp = join(x,z)

    h1 = T.nnet.relu(ln(T.dot(inp, p['w1']) + p['b1']),alpha=0.02)
    h2 = T.nnet.relu(ln(T.dot(h1,p['w2']) + p['b2']),alpha=0.02)
    h3 = T.dot(h2,p['w3']) + p['b3']

    y = h3

    return y

def net_d(p,zl,zn,x):

    #(z,z) -> 1

    inp = join3(zl,zn,x)


    h1 = T.nnet.relu(wn(T.dot(inp, p['w1']), p['w1']) + p['b1'],alpha=0.02)
    #h1 *= T.cast(srng.binomial(n=1,p=0.5,size=h1.shape),'float32')
    h2 = T.nnet.relu(wn(T.dot(h1,p['w2']), p['w2']) + p['b2'],alpha=0.02)
    #h2 *= T.cast(srng.binomial(n=1,p=0.5,size=h2.shape),'float32')
    h3 = T.dot(h2,p['w3']) + p['b3']

    y = h3

    disc_val = y.mean()

    return disc_val, h2

'''
    Take an x and zl, then map to zn, then map to x.   
'''
def waking(params_q, params_p, x, zl):

    znext = net_q(params_q,x,zl) + zl

    xrec = net_p(params_p,znext)

    zrec = net_zr(params_zr,znext)

    waking_loss = T.abs_(x - xrec).mean() + T.abs_(zl - zrec).mean() + 0.1 * T.abs_(znext - zl).mean()

    return znext, xrec, waking_loss

def sleep(params_s, params_p, zl, step):

    zn = net_s(params_s,zl,step) + zl

    xg = net_p(params_p,zn)

    sleep_loss = 0.0

    return zn, xg, sleep_loss

if __name__ == "__main__":


    x = T.matrix()
    zl_waking = T.matrix()
    zl_sleep = T.matrix()
    step = T.iscalar()
    params_p = init_params_p()
    params_q = init_params_q()
    params_d = init_params_d()
    params_s = init_params_s()
    params_zr = init_params_zr()

    zn_waking, xrec, waking_loss = waking(params_q, params_p, x, zl_waking)
    zn_sleep, xg, sleep_loss = sleep(params_s,params_p, zl_sleep,step)

    #zn_waking = -2.0 + 0.1*srng.normal(size=zn_waking.shape) + 0.0*zn_waking

    #Should we reweight z down?  

    disc_waking, h_waking = net_d(params_d, zl_waking*0.1, zn_waking*0.1, x)

    #zn_sleep = -2.0 + 0.0*srng.normal(size=zl_sleep.shape) + 0.0*zn_sleep
    
    disc_sleep, h_sleep = net_d(params_d, zl_sleep*0.1, zn_sleep*0.1, xg)

    matching_loss = T.abs_(h_waking.mean(axis=0) - h_sleep.mean(axis=0)).sum()

    print "DISC USING sgd 0.0001"

    disc_updates = clip_updates(lasagne.updates.sgd(disc_sleep - disc_waking, params_d.values(), learning_rate=0.0001), params_d.values())
    #waking_updates = lasagne.updates.adam(waking_loss + 0.1 * matching_loss + 0.1 * disc_waking, params_q.values() + params_p.values() + params_zr.values(), beta1=0.5)
    #sleep_updates = lasagne.updates.adam(0.1 * matching_loss - 0.1 * disc_sleep, params_s.values() + params_p.values(), beta1=0.5)

    l2_penalty = 0.0 * T.sqrt(T.sum(T.sqr(zn_waking))) + 0.0 * T.sqrt(T.sum(T.sqr(zn_sleep)))

    gen_updates = lasagne.updates.adam(l2_penalty + waking_loss + 0.01 * matching_loss + 0.0 * disc_waking - 0.0 * disc_sleep, params_q.values() + params_p.values() + params_zr.values() + params_s.values(), beta1=0.5)


    #need to add parameter clipping.  

    #x,zl -> zn -> x_rec.  
    #waking_train = theano.function([x, zl_waking], outputs = [waking_loss, zn_waking, xrec], updates = waking_updates)

    #zl -> zn
    #sleep_train = theano.function([zl_sleep,step], outputs =[zn_sleep,xg], updates = sleep_updates)

    gen_train = theano.function([x, zl_waking, zl_sleep, step], outputs = [waking_loss, zn_waking, xrec, zn_sleep, xg, matching_loss], updates = gen_updates)

    disc_train = theano.function([x, zl_waking, zl_sleep,step], outputs = [disc_waking, disc_sleep], updates = disc_updates)


    for i in range(0,85000):
        #Construct x.  

        xl = []

        for j in range(0,8):
            xl.append(np.random.normal([-0.5,-0.5,0.0,0.0,-0.5],[0.01,0.01,0.01,0.01,.01],size=5))
            xl.append(np.random.normal([0.5,0.5,0.0,0.0,0.5],[0.01,0.01,0.01,0.01,0.01],size=5))

        xseq = np.vstack(xl).astype('float32')

        #print "xseq shape", xseq.shape

        z_waking = np.zeros(shape=(16,1024)).astype('float32')
        z_sleep = np.zeros(shape=(16,1024)).astype('float32')
        #z_sleep = rng.normal(size=(16,1024)).astype('float32')

        max_step=5
        for step in range(0,5):
            z_sleep_last = z_waking
            z_waking_last = z_waking

            waking_loss, z_waking, xrec, z_sleep, xg, matching_loss = gen_train(xseq[:,step:step+1], z_waking_last, z_sleep_last,step)

            for j in range(5):
                d_waking, d_sleep = disc_train(xseq[:,step:step+1],z_waking_last,z_sleep_last,step)

            if i % 100 == 0:
                print "Disc should pull z_waking towards zero"
                print "iteration", i
                print "step", step
                print "True",xseq[:,step:step+1]
                print "Rec", xrec
                print "xg", xg
                print "z waking min mean max", z_waking.min(), z_waking.mean(), z_waking.max()
                print "z sleep min mean max", z_sleep.min(), z_sleep.mean(), z_sleep.max()
                print "d_waking", d_waking
                print "d_sleep", d_sleep
                print "waking loss", waking_loss
                print "matching loss", matching_loss



