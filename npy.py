import numpy as np
import itertools
import functools

def prime_stream():
    stream = itertools.count(2)
    sieve = lambda x, y: x % y != 0
    while True:
        prime = next(stream)
        stream = filter(functools.partial(sieve, y=prime), stream)
        yield prime

# 
prime_gen = prime_stream()
sampledata1 = np.reshape(np.array([next(prime_gen) for _ in range(1)], dtype=np.float32), (1,))
del prime_gen

# 
prime_gen = prime_stream()
sampledata2 = np.empty((2,2), dtype=np.complex64)
for i in range(2):
    for j in range(2):
        sampledata2[i][j] = next(prime_gen)
sampledata2 = np.reshape(sampledata2, (4,))
del prime_gen

# 
prime_gen = prime_stream()
sampledata3 = np.empty((152, 2, 4), dtype=np.float32)
for l in range(152):
    for i in range(2):
        for j in range(4):
            sampledata3[l][i][j] = next(prime_gen)
sampledata3 = np.reshape(sampledata3, (152*2*4,))
del prime_gen

# store sampledataN
with open("sampledata1.b", 'wb') as f:
    f.write(sampledata1)
np.save('sampledata1.npy', sampledata1)

with open('sampledata2.b', 'wb') as f:
    f.write(sampledata2)
np.save('sampledata2.npy', sampledata2)

with open('sampledata3.b', 'wb') as f:
    f.write(sampledata3)
np.save('sampledata3.npy', sampledata3)

# print
print(sampledata1)
print(sampledata2)
print(sampledata3)