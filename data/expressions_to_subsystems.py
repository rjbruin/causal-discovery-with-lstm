'''
Created on 29 mrt. 2017

@author: Robert-Jan
'''
import os;

if __name__ == '__main__':
    source = 'expressions_positive_integer_answer_deep';
    target = 'subsystems_deep';
    
    f = open(os.path.join(source, 'all.txt'));
    f_out = open(os.path.join(target, 'all.txt'), 'w');
    
    for line in f:
        f_out.write(line.strip() + ";0\n");
    
    f.close();
    f_out.close();