with open('resultlabelres50.txt') as f:
    lines = f.readlines()

with open('/home/yuchaohui/ych/caffe_ych/ssssss.txt') as ff:
    lines2 = ff.readlines()

pool = []
for li in lines: 
    li = li[:-1]
    pool.append(li.split(' ')[0].split('/')[-1])
print len(list(set(pool)))

count = 0
for li in lines2:
    li = li[:-1]
    if li in pool:
	count += 1
print count

