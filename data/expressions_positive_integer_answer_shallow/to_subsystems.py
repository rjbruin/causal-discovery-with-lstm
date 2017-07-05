f = open('all.txt','r');
f_out = open('all_subs.txt','w');

for line in f:
    f_out.write(line.strip() + ";0\n");

f.close();
f_out.close();
