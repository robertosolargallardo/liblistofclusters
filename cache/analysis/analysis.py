import numpy as np
import sys
f=open(sys.argv[1],'r')
IS=[[float(num) for num in line.strip().split(' ')] for line in f]

f=open(sys.argv[2],'r')
CS=[[float(num) for num in line.strip().split(' ')] for line in f]

for i in range(0,len(IS)):
	rem=CS[i][len(CS[i])-1]/IS[i][len(IS[i])-1]-1.0
	res=sum(CS[i][1:])/sum(IS[i][1:])-1.0

	print rem,res
