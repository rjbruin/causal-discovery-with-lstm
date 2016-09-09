'''
Created on 8 sep. 2016

@author: Robert-Jan
'''
from _collections import defaultdict

import numpy as np;

class ExpressionsByPrefix(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.expressions = [];
        self.prefixedExpressions = defaultdict(ExpressionsByPrefix);
    
    def add(self, expression):
        self.expressions.append(expression);
        if (len(expression) > 1):
            prefix = expression[0];
            self.prefixedExpressions[prefix].add(expression[1:]);
    
    def _get(self, prefix, level):
        if (level == 0):
            return (None, self.expressions);
        
        return (prefix, self.prefixedExpressions[prefix[0]].get(prefix[1:], level-1));
    
    def get(self, prefix, level):
        if (level == 0):
            return self.expressions;
        
        return map(lambda suffix: prefix[0] + suffix, self.prefixedExpressions[prefix[0]].get(prefix[1:], level-1));
    
    def get_random(self, max_size):
        nrExpressions = len(self.expressions);
        if (nrExpressions <= max_size):
            return self.expressions;
        else:
            availablePrefixes = self.prefixedExpressions.keys();
            if (len(availablePrefixes) == 0):
                raise ValueError("No prefixed available!");
            prefix = np.random.randint(0,len(availablePrefixes));
            return map(lambda suffix: availablePrefixes[prefix] + suffix, self.prefixedExpressions[availablePrefixes[prefix]].get_random(max_size));
    
#     def get(self, prefix):
#         level = len(prefix);
#         if (level == 0):
#             return "";
#         
#         answers = self._get(prefix, level);
#         for _ in range(len(prefix)):
#             answers = answers[1];
#         
#         expressions = [prefix];
#         while (answers[0] != None):
#             expressions = map(lambda suffix: )
            
            
            
            
            
            