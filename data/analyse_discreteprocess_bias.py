'''
Created on 12 apr. 2017

@author: Robert-Jan
'''

def fileNgrams(filename, nmax):
    f = open(filename);
    storage = {k: {} for k in range(2,nmax+1)}
    for line in f:
        line = line.strip().split(";")[1];
        storage = ngrams(line, storage, nmax);
    f.close();
    return storage;

def ngrams(line, storage, nmax):
    for n in range(2,nmax+1):
        for i in range(n,len(line)-n):
            if (line[i:i+n] not in storage[n]):
                storage[n][line[i:i+n]] = 0;
            storage[n][line[i:i+n]] += 1;
    return storage;