# coding: utf-8

import matplotlib.pylab as pylab
import numpy as np

class RungeKutta(object):
	
	def __init__(self, function_sys, init_cond, step = 0.01):
		'''
		<function_sys>	array of functions of same input type func(x, x_1, ... x_m)
						example for y''+3y'+2y=x :
						[lambda x,y1,y2: y2
						lambda x,y1,y2: x-3*y2-2*y1]
		<init_cond>		array of init condition of each function
		'''
		assert len(function_sys) == (len(init_cond)-1)
		self.func = function_sys
		self.cond = init_cond
		self.Y = [init_cond[1]]
		self.X = [0]

		self.step = step
		self.k = [[], [], [], []]

	def NextStep(self, step = None):
		'''
		Single step of Runge-Kutta

		<step> 	len of single Runge-Kutta step
		'''
		if step is None:
			step = self.step

		cond = self.cond[:]
		self.k[0] = [step] + list(map(lambda f:  step*f(*(cond)), self.func))
		
		for i in range(len(self.cond)):
			cond[i] = self.cond[i] + self.k[0][i]/2.0
		self.k[1] = [step] + list(map(lambda f:  step*f(*(cond)), self.func))

		for i in range(len(self.cond)):
			cond[i] = self.cond[i] + self.k[1][i]/2.0
		self.k[2] = [step] + list(map(lambda f:  step*f(*(cond)), self.func))

		for i in range(len(self.cond)):
			cond[i] = self.cond[i] + self.k[2][i]
		self.k[3] = [step] + list(map(lambda f:  step*f(*(cond)), self.func))

		for i in range(1, len(self.cond)):
			self.cond[i] +=  (self.k[0][i] + 2.0*self.k[1][i] + 2.0*self.k[2][i] + self.k[3][i])/6.0
		self.cond[0] += step
		
		return self.cond[:]
	
	def N_steps(self, N = None, end_point=None):
		if N is None:
			if end_point is not None:
				N = int(end_point/self.step)
			else:
				N = 1000
		if end_point is not None:
			self.step = end_point/N
		for n in range(N):
			cond = self.NextStep()
			self.Y.append(cond[1])
			self.X.append(cond[0])
	
	def plot(self, color = "r", style = '-', label = 'Runge-Kutta'):
		return pylab.plot(self.X, self.Y, color+style, label = label)



def test1():
	# y''+3y'+2y=1
	#W=(S^2+3S+2)^-1
	f = [
	lambda x,y,y1: y1, 			# diff(y)=y1
	lambda x,y,y1: 1-3*y1-2*y 	# diff2(y)=diff(y1)=1-3y1-2y
	]

	init_cond = [0, 0, 0] 		# X0, Y0, Z0

	s = RungeKutta(f,init_cond, step = 0.01)
	s.N_steps(2000,15)

	pylab.figure()
	s.plot()
	X = np.linspace(0, s.X[-1], 10000)
	pylab.plot(X, 0.5-np.exp(-X)+0.5*np.exp(-2*X), "b-", label = "Analitic solution")
	pylab.legend(loc = 'best')
	return s

def test2():
	# y''+y=0
	f = [
	lambda x,y,y1: y1,	# diff(y)=y1
	lambda x,y,y1: -y 	# diff2(y)=diff(y1)=-y
	]

	init_cond = [0, 0, 2] 		# X0, Y0, Z0

	pylab.figure()
	s = RungeKutta(f,init_cond, step = 0.1)
	s.N_steps(2000,15)
	s.plot()
	X = np.linspace(0, s.X[-1], 10000)
	pylab.plot(X, 2*np.sin(X), "b-", label = "Analitic solution")
	pylab.legend(loc = 'best')
	return s

if __name__ == '__main__':
	s1 = test1()
	s2 = test2()
	pylab.show()
