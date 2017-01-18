import gzip
import cPickle as pickle

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid

trainy = trainy.astype('int32')
validy = validy.astype('int32')

def get():
    r = random.randint(0,40000)
    return trainx[r:r+16]


