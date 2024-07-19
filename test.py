import debug
from fairseq2.data import read_iterator

class TestIterator:
	def __init__(self):
		self.reset()
	def reset(self):
		self.i = 0
	def __iter__(self):
		return self
	def __next__(self):
		self.i += 1
		return self.i
	def __getstate__(self):
		return self.i
	def __setstate__(self, i):
		self.i = i
		
@debug.traced
def main():
    pipeline = read_iterator(TestIterator()).and_return() 
    it = iter(pipeline)
    print(next(it))
    print(next(it))

if __name__ == "__main__":
    debug.reexecute_if_unbuffered()
    main()
