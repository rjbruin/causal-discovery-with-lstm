'''
Created on 13 sep. 2016

@author: Robert-Jan
'''

import sys;

from tools.arguments import processCommandLineArguments;
from tools.model import constructModels

if __name__ == '__main__':
    # Script parameters
    target_size = 10000;
    
    parameters = processCommandLineArguments(sys.argv[1:]);
    datasets, model = constructModels(parameters, 0, {});
    dataset = datasets[0];
    
    print(str(parameters));
    
    print(dataset.expressionLengths);
    print(dataset.testExpressionLengths);
    
    # Determine the minimum max_length needed to get batches quickly
#     min_samples_required = dataset.lengths[dataset.TRAIN] * 0.10;
#     max_length = model.n_max_digits;
#     samples_available = dataset.expressionLengths[max_length];
#     while (samples_available < min_samples_required):
#         max_length -= 1;
#         samples_available += dataset.expressionLengths[max_length];
#     
#     discoveredExpressions = {};
#     hits = 0;
#     expressionsDiscovered = 0;
#     while (expressionsDiscovered < target_size):
#         _, _, _, expressions, _, interventionLocation = get_batch(True, dataset, model, 5, max_length);
#         for exp in expressions:
#             if (exp in discoveredExpressions):
#                 hits += 1;
#             else:
#                 discoveredExpressions[exp] = True;
#                 expressionsDiscovered += 1;
#             
#     print("# Unique expressions: %d" % (expressionsDiscovered));
#     print("# Hits: %d" % (hits));
#     print("Sample size: %d" % (target_size));