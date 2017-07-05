if __name__ == '__main__':
    source = 'all.txt';
    f = open(source);

    histogram = {k: 0 for k in range(10)};

    for line in f:
        line = line.strip();
        equals = line.index('=');
        answersize = len(line[equals+1:]);
        histogram[answersize] += 1;

    print(histogram);
