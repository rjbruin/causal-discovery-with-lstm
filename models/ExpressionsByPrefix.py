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
        self.fullExpressions = [];
        self.expressions = [];
        self.prefixedExpressions = defaultdict(ExpressionsByPrefix);
        self.primedExpressions = [];
        self.maxExpressionSize = 0;
    
    def add(self, expression, expression_prime):
        self._add(expression, expression, expression_prime);
    
    def _add(self, expression, fullExpression, expression_prime):
        self.expressions.append(expression);
        self.fullExpressions.append(fullExpression);
        self.primedExpressions.append(expression_prime);
        
        if (len(expression) > self.maxExpressionSize):
            self.maxExpressionSize = len(expression);
        
        if (len(expression) > 0):
            prefix = expression[0];
            if (len(prefix) > 1):
                print(prefix);
            self.prefixedExpressions[prefix]._add(expression[1:], fullExpression, expression_prime);
    
    def get(self, prefix, getStructure=False):
        if (len(prefix) == 0):
            if (getStructure):
                return self;
            else:
                try:
                    index = self.expressions.index("");
                except ValueError:
                    return (False, False, self.fullExpressions, self.primedExpressions);
                return (self.fullExpressions[index], self.primedExpressions[index], 
                        self.fullExpressions, self.primedExpressions);
         
        return self.prefixedExpressions[prefix[0]].get(prefix[1:], getStructure=getStructure);
#     
#     def get_random(self, max_size):
#         nrExpressions = len(self.expressions);
#         if (nrExpressions <= max_size):
#             return self.expressions;
#         else:
#             availablePrefixes = self.prefixedExpressions.keys();
#             if (len(availablePrefixes) == 0):
#                 raise ValueError("No prefixed available!");
#             prefix = np.random.randint(0,len(availablePrefixes));
#             return map(lambda suffix: availablePrefixes[prefix] + suffix, self.prefixedExpressions[availablePrefixes[prefix]].get_random(max_size));
    
    def get_random_by_length(self, length, getStructure=False):
        if (length == 0):
            if (getStructure):
                return self;
            else:
                return (self.fullExpressions, self.primedExpressions);
        
        availablePrefixes = self.prefixedExpressions.keys();
        
        # DEBUG
        if (len(availablePrefixes) == 0):
            print("yo");
        
        prefix = np.random.randint(0,len(availablePrefixes));
        startingPrefix = prefix;
        
        while (self.prefixedExpressions[availablePrefixes[prefix]].maxExpressionSize < length-1):
            # or len(self.prefixedExpressions[availablePrefixes[prefix]].prefixedExpressions.keys()) == 0):
            prefix = (prefix + 1) % len(availablePrefixes);
            if (prefix == startingPrefix):
                # This branch has no prefixes with expressions
                if (getStructure):
                    return None;
                else:
                    return ([],[]);
        
        return self.prefixedExpressions[availablePrefixes[prefix]].get_random_by_length(length-1, getStructure=getStructure);
    
#     def count(self, prefix):
#         if (prefix == ""):
#             return len(self.candidates);
#         return self.prefixedExpressions[prefix[0]].count(prefix[1:]);
#     
#     def countForPrefixes(self, prefix, allowedPrefixes):
#         if (prefix == ""):
#             count = 0;
#             for pref in allowedPrefixes:
#                 if (pref in self.prefixedExpressions):
#                     count += len(self.prefixedExpressions[pref].expressions);
#             return count;
#         return self.prefixedExpressions[prefix[0]].count(prefix[1:], allowedPrefixes);
#     
#     def countAll(self, debug=False):
#         return self._countAll(0, debug=debug);
    
#     def _countAll(self, level, debug=False):
#         thisLevel = len(filter(lambda e: len(e) == 1, self.expressions));
#         if (len(self.prefixedExpressions.keys()) == 0):
#             if (debug):
#                 print("Bottom reached at level %d" % level);
#             return [thisLevel];
#         # Get the lists of level counts for each prefix
#         # deeper contains #prefixes lists of flat integer lists with length
#         # equal to the longer expression in the branch 
#         deeper = map(lambda p: self.prefixedExpressions[p]._countAll(level+1, debug=debug), self.prefixedExpressions.keys())
#         # We fill out the shorter lists with zeros so we can map the sum
#         max_length = max(map(lambda d: len(d), deeper));
#         max_lengthed_deeper = [];
#         for d in deeper:
#             d.extend([0 for _ in range(max_length - len(d))]);
#             max_lengthed_deeper.append(d);
#         # Flatten using sum to obtain a flat list of lengths per level
#         levelSums = map(lambda *ls: sum(ls), *max_lengthed_deeper);
#         return [thisLevel] + levelSums;
    
    def exists(self, prefix):
        if (len(prefix) == 0):
            return "" in self.expressions;
        if (prefix[0] in self.prefixedExpressions):
            return self.prefixedExpressions[prefix[0]].exists(prefix[1:]);
        else:
            return False;
