import re

f = open('log')
lines = f.readlines()
f.close()
iter = []
acc1 = []
acc5 = []
for li in lines:
    li = li[:-1]
    tmp = re.findall(r'Iteration (\d+).*Testing', li)
    if tmp:
	iter.append(tmp)
    tmp1 = re.findall(r'accuracy@1 = (\d.*\d+)', li)
    if tmp1:
	acc1.append(tmp1)
    tmp5 = re.findall(r'accuracy@5 = (\d.*\d+)', li)
    if tmp5:
        acc5.append(tmp5)

f = open('itertest','a+')
for i in xrange(50):
    f.write(str(iter[i][0])+' '+str(acc1[i][0])+' '+str(acc5[i][0])+'\n')
f.close()




