import matplotlib.pyplot as plt


with open('iterloss') as f:
    lines = f.readlines()
iter = []
loss = []
for li in lines:
    li = li[:-1].strip().split(' ')
    iter.append(int(li[0]))
    loss.append(float(li[1]))
plt.plot(iter,loss)
plt.title("resnet50 Loss")
plt.legend(('Loss',), loc='upper right')

"""
with open('itertest') as f:
    lines = f.readlines()
iter = []
acc1 = []
acc5 = []
for li in lines:
    li = li[:-1].strip().split(' ')
    iter.append(int(li[0]))
    acc1.append(float(li[1]))
    acc5.append(float(li[2]))
plt.plot(iter,acc1,'r')
plt.plot(iter,acc5,'b')
plt.title("resnet50 test acc")
plt.legend(('acc1','acc3'), loc='upper right')
"""
plt.show()







