'''
Created on 22 feb. 2017

@author: Robert-Jan
'''

from scipy.stats import ttest_ind;

if __name__ == '__main__':
    values1 = map(float,raw_input("Values for sequence 1 (separated by ,): ").split(","));
    values2 = map(float,raw_input("Values for sequence 2 (separated by ,): ").split(","));
    
    stat, pvalue = ttest_ind(values1, values2, equal_var=True);
    
    print("T-statistic: %.8f" % stat);
    print("Two-tailed p-value: %.8f" % pvalue);