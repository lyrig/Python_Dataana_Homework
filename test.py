'''
Date: 2023-02-27 09:43:07
LastEditTime: 2023-02-27 09:44:37
'''
def fab(max=5):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n +=1

i = fab(5)
for t in i:
    print(t)