# -*- coding: utf-8 -*-

import numpy as np

tc = []
td = []

for i in range(10):
	ans = [0,0,0,0,0,0,0,0,0,0]
	ans[i] = 1
	filename = "fb80txt/"+str(i)+".fb80txt"
	f = open(filename, 'r')
	print(filename+"을 합친다.")
	for line in f:
		line = line.strip()
		s_n = line.split() # string type으로 숫자가 쓰여있는 배열
		narr = []
		for numb in s_n: # string을 다 숫자로 바꿔서 narr에 넣어준다.
			narr.append(float(numb))
		tc.append(narr)
		td.append(ans)
	f.close

# nd array로 변환
tc = np.array(tc)
td = np.array(td)
tc = tc.astype(np.float32)
td = td.astype(np.float32)
f.close()

np.savez("mydata", X=tc, Y=td)

