'''
Created on 20 dec. 2016

@author: Robert-Jan
'''

from models.GeneratedExpressionDataset import GeneratedExpressionDataset;

def testEffectMatcher():
    print("Preloading data...");
    dataset = GeneratedExpressionDataset('./data/seq2ndmarkov_both_2/all.txt', './data/seq2ndmarkov_both_2/test.txt', './data/seq2ndmarkov_both_2/config.json',
                                         operators=2,
                                         digits=8,
                                         dataset_type=GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV,
                                         bothcause=True,
                                         finishExpressions=True);
    
    print("Starting test...");
    for i in range(len(dataset.expressionsByPrefix.expressions)):
        if (i % 1000):
            print("%.2f%% done" % (i / 10000.));
        
        top = dataset.strToIndices(dataset.expressionsByPrefix.expressions[i]);
        bot = dataset.strToIndices(dataset.expressionsByPrefix.primedExpressions[i]);
        
        test = dataset.effect_matcher_seq2ndmarkov_both_2(top, bot, 8, 2, True);
        if (not test):
            print("(effect) %d is wrong: top = %s, bot = %s" % (i, dataset.expressionsByPrefix.expressions[i], dataset.expressionsByPrefix.primedExpressions[i]));

if __name__ == '__main__':
    testEffectMatcher();