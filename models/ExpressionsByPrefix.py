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
    
    PREFIXES = [str(i) for i in range(10)] + ['+','-','*','/','(',')','='];

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
    
    def get(self, prefix, getStructure=False, safe=False):
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
        if (safe and prefix[0] not in self.prefixedExpressions):
            return False; 
        return self.prefixedExpressions[prefix[0]].get(prefix[1:], getStructure=getStructure, safe=safe);
    
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
    
    def get_next(self, lastPath):
        availablePrefixes = sorted(self.prefixedExpressions.keys());
        if (len(lastPath) > 0):
            # If the path is not empty we try the path instructions
            pathType, index = lastPath[0];
            if (pathType == 0):
                if (index >= len(availablePrefixes)):
                    # whut
                    pass;
                result = self.prefixedExpressions[availablePrefixes[index]].get_next(lastPath[1:]);
                if (result != False):
                    return result[0], [(0, index)] + result[1];
                # If the path instructions fail we fall through to iterating over
                # the options at the current level
        else:
            # If there is no path left we start iterating over the options at
            # the current level
            pathType = 0;
            index = -1;
            lastPath = [(0, 0)];
        
        if (pathType == 0):
            # We start with checking the prefixes for options if 1) we don't
            # have path instructions or 2) the path says the last option was
            # a prefix
            if (len(availablePrefixes) > 0):
                success = False;
                currentIndex = index;
                while (not success):
                    # Iterate over all prefixes until we find a valid one
                    currentIndex += 1;
                    if (currentIndex >= len(self.prefixedExpressions)):
                        # We ran out of valid prefixes to explore
                        break;
                    # Call without path because we are changing the path branch 
                    # we follow
                    result = self.prefixedExpressions[availablePrefixes[currentIndex]].get_next([]);
                    if (result == False):
                        # The explored prefix has no valid options left
                        continue;
                    return result[0], [(0, currentIndex)] + result[1];
            # If we need to explore the expressions after considering the
            # prefixes we need to reset the index as it was supposed to point
            # to a prefix, not an expression
            index = -1;
        
        # Find the next expression
        expressionsAtCurrentLevel = filter(lambda e: e[1] == "", enumerate(self.expressions));
        choiceExpression = index + 1;
        if (choiceExpression >= len(expressionsAtCurrentLevel)):
            # Expressions are done as well
            return False;
        return self.fullExpressions[expressionsAtCurrentLevel[choiceExpression][0]], [(1, choiceExpression)];
    
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
