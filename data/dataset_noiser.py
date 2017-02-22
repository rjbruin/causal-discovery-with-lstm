'''
Created on 8 nov. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;

if __name__ == '__main__':
    source = sys.argv[1];
    target = sys.argv[2];
    triggerProb = 0.01;
    ignoreSymbols = [';'];
    symbols = [str(i) for i in range(10)] + ['+','-','*','/','(',')','='];
    
    sources = ["/all.txt"];
    
    for part in sources:
        s = source + part;
        t = target + part; 
        
        f = open(s);
        f_target = open(t, 'w');
        
        line = f.readline().strip();
        while (line != ""):
            probs = np.random.random((len(line)));
            newLine = "";
            for i, s in enumerate(line):
                if (probs[i] <= triggerProb and s not in ignoreSymbols):
                    newSymbol = symbols[np.random.randint(len(symbols))];
                    newLine += newSymbol;
                else:
                    newLine += line[i];
            f_target.write(newLine + '\n');
            line = f.readline().strip();
        
        f.close();
        f_target.close();
    
    print("Done!");