import random

def monte_carlo_pi(n):
    inside_circle = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4*inside_circle/n

n = 30000000
pi = monte_carlo_pi(n)
print("PI is approximately", pi)
