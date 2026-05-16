import numpy as np
import sys
f=open(sys.argv[1],'r')
IS=[[float(num) for num in line.strip().split(' ')] for line in f]

f=open(sys.argv[2],'r')
CS=[[float(num) for num in line.strip().split(' ')] for line in f]

f=open(sys.argv[3],'r')
DATA=[[float(num) for num in line.strip().split(' ')] for line in f]


for i in range(0,len(IS)):
	rem=CS[i][len(CS[i])-1]/IS[i][len(IS[i])-1]-1.0
	res=sum(CS[i][1:])/sum(IS[i][1:])-1.0

	s=0
	for j in range(0,len(DATA[i]),3):
		s+=DATA[i][j+2]*(DATA[i][j]/DATA[i][j+1])	
		break

	if rem < 0:
		print IS[i]
		print CS[i]

	print rem,res,s
