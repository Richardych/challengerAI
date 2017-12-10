
with open('val.txt','rw') as f:
    lines = f.readlines()
with open('valbk.txt','w') as ff:
    for li in lines:
        li = li[:-1]
        ff.write(li[0]+'/'+li+'\n')
        print li[0],'/',li
