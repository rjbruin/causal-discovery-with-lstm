'''
Created on 8 sep. 2016

@author: Robert-Jan
'''
from _collections import defaultdict
import numpy as np;

class SequencesByPrefix(object):
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
        self.prefixedExpressions = defaultdict(SequencesByPrefix);
        self.primedExpressions = [];
        self.maxExpressionSize = 0;
        self.nearest_cache = {};
    
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
            self.prefixedExpressions[prefix]._add(expression[1:], fullExpression, expression_prime);
    
    def get(self, prefix, getStructure=False, safe=False, alsoGetStructure=False):
        if (len(prefix) == 0):
            if (getStructure):
                return self;
            else:
                try:
                    index = self.expressions.index("");
                except ValueError:
                    if (not alsoGetStructure):
                        return (False, False, self.fullExpressions, self.primedExpressions);
                    else:
                        return (False, False, self.fullExpressions, self.primedExpressions, self);
                if (not alsoGetStructure):
                    return (self.fullExpressions[index], self.primedExpressions[index], 
                            self.fullExpressions, self.primedExpressions);
                else:
                    return (self.fullExpressions[index], self.primedExpressions[index], 
                            self.fullExpressions, self.primedExpressions, self);
        if (safe and prefix[0] not in self.prefixedExpressions):
            return False; 
        return self.prefixedExpressions[prefix[0]].get(prefix[1:], getStructure=getStructure, safe=safe, alsoGetStructure=alsoGetStructure);
    
    def get_closest(self, target):
        # Go down as far as possible into the given target
        # Where it starts to deviate, choose a target that is 
        if (len(target) == 0):
            if ("" in self.expressions):
                return self.get(target);
        elif (target[0] in self.prefixedExpressions):
            return self.prefixedExpressions[target[0]].get_closest(target[1:]);
        
        if (target not in self.nearest_cache):
        
            nearest = -1;
            nearest_score = 1000000;
            
            exprSize = len(self.expressions);
            
#             limit = 1000;
#             sample = False;
#             if (exprSize > limit):
#                 sample = True;
#             else:
#                 limit = exprSize;
#             samples = {};
#             if (sample):
#                 for _ in range(10):
#             for j in range(limit):
#                 if (sample):
#                     i = np.random.randint(0,exprSize);
#                 else:
#                     i = j;
#                 expr = self.expressions[i];
#                 current_score = SequencesByPrefix.string_difference(target, expr);
#                 if (current_score < nearest_sampled_score):
#                     nearest = i;
#                     nearest_score = current_score;
                
            for j in range(exprSize):
                expr = self.expressions[j];
                current_score = SequencesByPrefix.string_difference(target, expr);
                if (current_score < nearest_score):
                    nearest = j;
                    nearest_score = current_score;
            
            self.nearest_cache[target] = nearest;
        else:
            nearest = self.nearest_cache[target];
        
        return (self.fullExpressions[nearest], self.primedExpressions[nearest], 
                self.fullExpressions, self.primedExpressions);
#         return (self.fullExpressions[nearest], self.primedExpressions[nearest], 
#                 self.fullExpressions, self.primedExpressions);
    
    def get_random_by_length(self, length, getStructure=False):
        if (length == 0):
            if (getStructure):
                return self;
            else:
                return (self.fullExpressions, self.primedExpressions);
        
        availablePrefixes = self.prefixedExpressions.keys();
        
        # DEBUG
        if (len(availablePrefixes) == 0):
            raise ValueError("BUG! This cannot occur of the maxExpressionSize check works right.");
        
        prefix = np.random.randint(0,len(availablePrefixes));
        startingPrefix = prefix;
        
        while (self.prefixedExpressions[availablePrefixes[prefix]].maxExpressionSize < length-1):
            prefix = (prefix + 1) % len(availablePrefixes);
            if (prefix == startingPrefix):
                # This branch has no prefixes with expressions
                if (getStructure):
                    return None;
                else:
                    return ([],[]);
        
        return self.prefixedExpressions[availablePrefixes[prefix]].get_random_by_length(length-1, getStructure=getStructure);
    
    
    
#     def get_next(self, lastPath):
#         availablePrefixes = sorted(self.prefixedExpressions.keys());
#         if (len(lastPath) > 0):
#             # If the path is not empty we try the path instructions
#             pathType, index = lastPath[0];
#             if (pathType == 0):
#                 if (index >= len(availablePrefixes)):
#                     # whut
#                     pass;
#                 result = self.prefixedExpressions[availablePrefixes[index]].get_next(lastPath[1:]);
#                 if (result != False):
#                     return result[0], [(0, index)] + result[1];
#                 # If the path instructions fail we fall through to iterating over
#                 # the options at the current level
#         else:
#             # If there is no path left we start iterating over the options at
#             # the current level
#             pathType = 0;
#             index = -1;
#             lastPath = [(0, 0)];
#         
#         if (pathType == 0):
#             # We start with checking the prefixes for options if 1) we don't
#             # have path instructions or 2) the path says the last option was
#             # a prefix
#             if (len(availablePrefixes) > 0):
#                 success = False;
#                 currentIndex = index;
#                 while (not success):
#                     # Iterate over all prefixes until we find a valid one
#                     currentIndex += 1;
#                     if (currentIndex >= len(self.prefixedExpressions)):
#                         # We ran out of valid prefixes to explore
#                         break;
#                     # Call without path because we are changing the path branch 
#                     # we follow
#                     result = self.prefixedExpressions[availablePrefixes[currentIndex]].get_next([]);
#                     if (result == False):
#                         # The explored prefix has no valid options left
#                         continue;
#                     return result[0], [(0, currentIndex)] + result[1];
#             # If we need to explore the expressions after considering the
#             # prefixes we need to reset the index as it was supposed to point
#             # to a prefix, not an expression
#             index = -1;
#         
#         # Find the next expression
#         expressionsAtCurrentLevel = filter(lambda e: e[1] == "", enumerate(self.expressions));
#         choiceExpression = index + 1;
#         if (choiceExpression >= len(expressionsAtCurrentLevel)):
#             # Expressions are done as well
#             return False;
#         return self.fullExpressions[expressionsAtCurrentLevel[choiceExpression][0]], [(1, choiceExpression)];
    
    def exists(self, prefix):
        if (len(prefix) == 0):
            return "" in self.expressions;
        if (prefix[0] in self.prefixedExpressions):
            return self.prefixedExpressions[prefix[0]].exists(prefix[1:]);
        else:
            return False;
        
    @staticmethod
    def string_difference(string1, string2):
        # Compute string difference
        score = 0;
        string1len = len(string1);
        k = 0;
        for k,s in enumerate(string2):
            if (string1len <= k):
                score += 1;
            elif (s != string1[k]):
                score += 1;
        score += max(0,len(string1) - (k+1));
        return score;
