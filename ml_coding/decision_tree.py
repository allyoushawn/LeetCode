import math

"""
entropy: Sum_i p_i log_2(p_i)
"""


def calculate_entropy(Y):
	val_list = set(Y)
	ret = 0
	eps = 10**(-15)
	for y in val_list:
		prob = sum([ val == y for val in Y]) / len(Y)
		ret += prob * math.log(prob + eps) / math.log(2)
	return -1 * ret


def information_gain(Y, Y1, Y2):
	p1 = len(Y1) / len(Y)
	p2 = len(Y2) / len(Y)

	h_y = calculate_entropy(Y)
	h_y1 = calculate_entropy(Y1)
	h_y2 = calculate_entropy(Y2)

	return h_y - (p1 * h_y1 + p2 * h_y2)

#print(calculate_entropy([0, 1]))

Y1 = [0,0,0,0,0,0,0,1]
Y2 = [0,0,0,0,0,0,1,1,1,1,1,1]
print(information_gain(Y1+Y2, Y1, Y2))
