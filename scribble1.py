import unittest

class Test1(unittest.TestCase):
	def test_1(self):
		self.assertTrue(True)
	def test_2(self):
		self.assertTrue(True)
class Test2(unittest.TestCase):
	def test_1(self):
		self.assertTrue(True)
if __name__ == '__main__':
	unittest.main()
