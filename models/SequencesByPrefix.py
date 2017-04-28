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
        
        self.noEditMarker = False;
    
    def add(self, expression, expression_prime):
        if (self.noEditMarker):
            raise ValueError("Cannot edit!");
        self._add(expression, expression, expression_prime);
    
    def _add(self, expression, fullExpression, expression_prime):
        if (self.noEditMarker):
            raise ValueError("Cannot edit!");
        self.expressions.append(expression);
        self.fullExpressions.append(fullExpression);
        self.primedExpressions.append(expression_prime);
        
        if (len(expression) > self.maxExpressionSize):
            self.maxExpressionSize = len(expression);
        
        if (len(expression) > 0):
            prefix = expression[0];
            self.prefixedExpressions[prefix]._add(expression[1:], fullExpression, expression_prime);
    
    def get(self, prefix, getStructure=False, safe=False, alsoGetStructure=False, filterExpressionPrime=None):
        if (len(prefix) == 0):
            if (getStructure):
                return self;
            else:
                candidates = self.expressions;
                fullCandidates = self.fullExpressions;
                primedCandidates = self.primedExpressions;
                # Filter for expression prime if given
                if (filterExpressionPrime is not None):
                    fullCandidates = [];
                    primedCandidates = [];
                    for i in range(len(self.fullExpressions)):
                        if (self.primedExpressions[i][:len(filterExpressionPrime)] == filterExpressionPrime):
                            candidates.append(self.expressions[i]);
                            fullCandidates.append(self.fullExpressions[i]);
                            primedCandidates.append(self.primedExpressions[i]);
                
                try:
                    index = candidates.index("");
                except ValueError:
                    if (not alsoGetStructure):
                        return (False, False, fullCandidates, primedCandidates);
                    else:
                        return (False, False, fullCandidates, primedCandidates, self);
                
                if (not alsoGetStructure):
                    return (fullCandidates[index], primedCandidates[index], 
                            fullCandidates, primedCandidates);
                else:
                    return (fullCandidates[index], primedCandidates[index], 
                            fullCandidates, primedCandidates, self);
        if (safe and prefix[0] not in self.prefixedExpressions):
            return False;
        
        return self.prefixedExpressions[prefix[0]].get(\
                prefix[1:], 
                getStructure=getStructure, 
                safe=safe, alsoGetStructure=alsoGetStructure,
                filterExpressionPrime=filterExpressionPrime);
    
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
    
    def get_random(self):
        candidate = np.random.randint(0,len(self.expressions));
        expression = self.expressions[candidate];
        return self.get(expression);
    
    def get_next(self, lastPath):
        """
        Iterates over all expressions in the storage in depth-first way.
        Path is a list of tuples (instruction, index).
        Instructions are:
        - 0 = go down to prefix 'index'
        - 1 = return expression at 'index' + 1.
        """
        self.noEditMarker = True;
        # Move along the path as far as possible
        availablePrefixes = sorted(self.prefixedExpressions.keys());
        if (len(lastPath) > 0):
            # If the path is not empty we try the path instructions
            pathType, index = lastPath[0];
            if (pathType == 0):
                if (index >= len(availablePrefixes)):
                    raise ValueError("whut: %d, %d" % (len(availablePrefixes), index));
                result = self.prefixedExpressions[availablePrefixes[index]].get_next(lastPath[1:]);
                if (result != False):
                    return result[0], [(0, index)] + result[1];
                # If the path instructions fail we fall through to iterating over
                # the options at the current level
        else:
            # If there is no path (left) we start iterating over the options at
            # the current level
            pathType = 0;
            index = -1;
        
        # At this point we have moved along the path as far as possible
        # pathType and index are the last instuctions in the path 
        # If pathType == 0 we are instructed to try prefixes at this level
        # This matches two scenario's: 
        # 1) we don't have path instructions or 
        # 2) the path says the last option was a prefix
        if (pathType == 0):
            if (len(availablePrefixes) > 0):
                currentIndex = index + 1;
                # Iterate over all prefixes until we find a valid one
                while (currentIndex < len(availablePrefixes)):
                    result = self.prefixedExpressions[availablePrefixes[currentIndex]].get_next([]);
                    if (result is not False):
                        return result[0], [(0, currentIndex)] + result[1];
                    # If result == False, the explored prefix has no valid options 
                    # left so we skip it
                    currentIndex += 1;
            # We found no suitable prefixes left at this level
            # We need to explore the expressions
            pathType = 1;
            # We need to reset the index as it was supposed to point
            # to a prefix, not an expression
            index = -1;
         
        # If we get to this point we have processed all prefixes at this and deeper levels
        # What remains is to try the expressions at this level
        expressionsAtCurrentLevel = filter(lambda e: e[1] == "", enumerate(self.expressions));
        choiceExpression = index + 1;
        if (choiceExpression >= len(expressionsAtCurrentLevel)):
            # Expressions are done as well
            return False;
        return self.fullExpressions[expressionsAtCurrentLevel[choiceExpression][0]], [(1, choiceExpression)];
    
    def exists(self, prefix, prime=None):
        if (len(prefix) == 0):
            if ("" in self.expressions):
                if (prime is None):
                    return True;
                else:
                    if (prime in self.primedExpressions):
                        return True;
            return False;
        if (prefix[0] in self.prefixedExpressions):
            return self.prefixedExpressions[prefix[0]].exists(prefix[1:], prime=prime);
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
