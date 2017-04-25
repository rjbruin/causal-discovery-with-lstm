'''
Created on 22 feb. 2017

@author: Robert-Jan
'''

from scipy.stats import ttest_ind;
import itertools;
from statsmodels.stats.power import tt_ind_solve_power;
import numpy as np;

if __name__ == '__main__':
#     values1 = map(float,raw_input("Values for sequence 1 (separated by ,): ").split(","));
#     values2 = map(float,raw_input("Values for sequence 2 (separated by ,): ").split(","));
    
    # Precisions
    values1 = [100.00, 95.26, 99.91, 100.00, 99.99];
    values2 = [85.8, 84.86, 85.87, 85.67, 85.87];
    
    # Losses
    values1 = [0.00007608, 0.02547330, 0.00128386, 0.00014547, 0.00016470];
    values2 = [0.07666870, 0.08185120, 0.07599110, 0.07889190, 0.07555600];
    
    rho_input = raw_input("Give values of rho to test (separated by ,): ").strip();
    if (rho_input == ''):
        rho_values = [len(values1)];
    else:
        rho_values = map(int,rho_input.split(","));
    
    for rho in rho_values:
        pvalues = [];
        print("Rho = %d:" % rho);
        for comb in itertools.combinations(range(len(values1)),rho): 
            vals1 = [values1[i] for i in comb];
            vals2 = [values2[i] for i in comb];
            stat, pvalue = ttest_ind(vals1, vals2, equal_var=True);
            power = tt_ind_solve_power(effect_size=(np.mean(vals1) - np.mean(vals2)) / np.std(vals1), nobs1=len(vals1), alpha=0.05, power=None);
            print("T-statistic / p / power: %.8f, %.8f, %s (%s, %s)" % (stat, pvalue, str(power), str(vals1), str(vals2)));
            pvalues.append(pvalue);
        print([p < 0.05 for p in pvalues]);
        print([p < 0.01 for p in pvalues]);
        print([p < 0.001 for p in pvalues]);
        print;