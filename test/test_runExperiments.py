'''
Created on 13 jun. 2016

@author: Robert-Jan
'''

if __name__ == '__main__':
    i = 0;
    while (i < 2):
        i += 1;
        i = i % 1000000;
        print(" " + str(i));