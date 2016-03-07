'''
Created on 7 mrt. 2016

@author: Robert-Jan
'''

import os;

if __name__ == '__main__':
    dataset = './expressions_positive_integer_answer_shallow';
    
    right_hands = {};
    
    f = open(os.path.join(dataset,'train.txt'));
    for line in f:
        args = line.split("=");
        right_hand = args[1];
        if (right_hand not in right_hands):
            right_hands[right_hand] = 0;
        right_hands[right_hand] += 1;
    
    f = open(os.path.join(dataset,'test.txt'));
    for line in f:
        args = line.split("=");
        right_hand = args[1];
        if (right_hand not in right_hands):
            right_hands[right_hand] = 0;
        right_hands[right_hand] += 1;
    
    print(sum(right_hands.values()));