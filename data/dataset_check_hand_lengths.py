if __name__ == '__main__':
    f = open('expressions_positive_integer_answer_shallow/all.txt');
    rightHandLengths = {};
    leftHandLengths = {};
    
    for line in f:
        left, right = line.split("=");
        right = right.strip();
        if (len(right) not in rightHandLengths):
            rightHandLengths[len(right)] = {};
        if (len(left) not in rightHandLengths[len(right)]):
            rightHandLengths[len(right)][len(left)] = 0;
        rightHandLengths[len(right)][len(left)] += 1;
        
        if (len(left) not in leftHandLengths):
            leftHandLengths[len(left)] = {};
        if (len(right) not in leftHandLengths[len(left)]):
            leftHandLengths[len(left)][len(right)] = 0;
        leftHandLengths[len(left)][len(right)] += 1;
    
    f.close();
    
    # Print
    for rl in range(7):
        if (rl in rightHandLengths):
            print("Right hand side length %d:" % rl);
            print(str(rightHandLengths[rl]));
    
    for ll in range(20):
        if (ll in leftHandLengths):
            print("Left hand side length %d:" % ll);
            print(str(leftHandLengths[ll]));