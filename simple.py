from fairseq2.data import read_sequence, read_iterator
from time import sleep

class TestIterator:
    def __init__(self):
        self.i = 0
    def reset(self):
        print(f"BEFORE self.i {self.i}")
        self.i = 0
        print(f"AFTER self.i {self.i}")
    def __iter__(self):
        return self
    def __next__(self):
        x = self.i
        self.i += 1
        return x
    def __getstate__(self):
        return self.i
    def __setstate__(self, i):
        self.i = i

pipeline = read_iterator(TestIterator()).and_return() 
#pipeline = read_sequence([1]).and_return() 
it = iter(pipeline)

print(next(it))
print(next(it))

pipeline.reset() 

print(next(it))
print(next(it))

del it
print("still ok??")
del pipeline
print("still ok?")
