import math
def countConsecutive():
	s = 0
	count = 0
	for i in range(63,3,-1):
		count += 1 
		print i, count
		choo = i*(i-1)*(i-2)/6.0
		s += (choo * count) 
	overall = 69*68*67*66*65/120.0
	return s/overall

print 1-countConsecutive()
	
