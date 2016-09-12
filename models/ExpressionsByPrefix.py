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
    
    def add(self, expression):
        self._add(expression, expression);
    
    def _add(self, expression, fullExpression):
        self.expressions.append(expression);
        self.fullExpressions.append(fullExpression);
        if (len(expression) > 1):
            prefix = expression[0];
            self.prefixedExpressions[prefix]._add(expression[1:], fullExpression);
    
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
    
    def get_random_by_length(self, length, getStructure=False):
        if (length == 0):
            if (getStructure):
                return self;
            else:
                return self.expressions;
        
        availablePrefixes = self.prefixedExpressions.keys();
        prefix = np.random.randint(0,len(availablePrefixes));
        startingPrefix = prefix;
        
        while (len(self.prefixedExpressions[availablePrefixes[prefix]].prefixedExpressions.keys()) == 0):
            prefix = (prefix + 1) % len(availablePrefixes);
            if (prefix == startingPrefix):
                # This branch has no prefixes with expressions
                if (getStructure):
                    return None;
                else:
                    return [];
        
        if (getStructure):
            return self.prefixedExpressions[availablePrefixes[prefix]].get_random_by_length(length-1, getStructure=getStructure);
        else:
            return map(lambda suffix: availablePrefixes[prefix] + suffix, self.prefixedExpressions[availablePrefixes[prefix]].get_random_by_length(length-1, getStructure=getStructure));
    
    def count(self, prefix):
        if (prefix == ""):
            return len(self.candidates);
        return self.prefixedExpressions[prefix[0]].count(prefix[1:]);
    
    def countForPrefixes(self, prefix, allowedPrefixes):
        if (prefix == ""):
            count = 0;
            for pref in allowedPrefixes:
                if (pref in self.prefixedExpressions):
                    count += len(self.prefixedExpressions[pref].expressions);
            return count;
        return self.prefixedExpressions[prefix[0]].count(prefix[1:], allowedPrefixes);
    
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
            
            
            
            
            
            