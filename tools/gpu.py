'''
Created on 21 jun. 2016

@author: Robert-Jan
'''

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy;

def using_gpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    #iters = 1000
    
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))  # @UndefinedVariable
    f = function([], T.exp(x))
    #for i in range(iters):
    #    r = f()
    return not numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]);

if __name__ == '__main__':
    print(using_gpu());