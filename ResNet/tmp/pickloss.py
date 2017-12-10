import re

f = open('logzscore.txt')
lines = f.readlines()
f.close()

iter = []
loss = []
f = open('iterloss','a+')
for li in lines:
    li = li[:-1]
    tmp = re.findall(r'Iteration (\d+).*loss = (\d.*\d+)', li)
    if tmp:
	print tmp[0][0],tmp[0][1]
	f.write(str(tmp[0][0])+' '+tmp[0][1]+'\n')
f.close()
"""
f = open('iterloss','a+')
for i in xrange(60):
    f.write(str(iter[i][0])+' '+str(acc1[i][0])+' '+str(acc3[i][0])+'\n')
"""

"""
iter = []
acc1 = []
acc3 = []
for li in lines:
    li = li[:-1]
    tmp = re.findall(r'Iteration (\d+).*Testing', li)
    if tmp:
	iter.append(tmp)
    tmp1 = re.findall(r'acc/top-1 = (\d.*\d+)', li)
    if tmp1:
	acc1.append(tmp1)
    tmp5 = re.findall(r'acc/top-5 = (\d.*\d+)', li)
    if tmp5:
        acc3.append(tmp5)

f = open('itertest','a+')
for i in xrange(60):
    f.write(str(iter[i][0])+' '+str(acc1[i][0])+' '+str(acc3[i][0])+'\n')
f.close()
"""



