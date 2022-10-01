# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:31:38 2020

@author: 王碧
"""


class A(object):
    def update(self):
        self.name = 'A';
        return self.name;
        
    def toString(self):
        return self.name;
    
class B(object):
    def update(self):
        self.name = 'B';
        return self.name;
        
    def doubleUpdate(self):
        self.update();
        self.name += '1';
        return self.name;
        
class C(A, B):
    def __init__(self):
        self.name = "C";
        self.v = A();
        
    def __getattr__(self, name):
        return getattr(self.v, name);
    
c = C();
print(c.update(), c.doubleUpdate())

class Minix1:
	"""该混合类为header列表末尾添加data1"""
	def get_header(self):
		print('run Minix1.get_header')
		ctx = super().get_header()
		ctx.append('data1')
		return ctx

class Minix2:
	"""该混合类为header列表头部添加data2"""
	def get_header(self):
		print('run Minix2.get_header')
		ctx = super().get_header()
		ctx.append('data2')
		return ctx

class Header:
	header = []
	def get_header(self):
		print('run Headers.get_header')
		return self.header if self.header else []


#python 的继承很神奇，节点构成是按照输入顺序，因此，后面类中的方法会被覆盖
class Final(Minix1, Minix2, Header):

	def get_header(self):
#		return super().get_header()
		return super(Final, self).get_header()
#		return Header.get_header(self)
    
print(Final.mro())
print(Final().get_header())