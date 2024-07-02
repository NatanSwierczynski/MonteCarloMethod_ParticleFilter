import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

N = 10_000
a, b = (50,50)
x_min, x_max = (0, 1)

randx = np.random.uniform(x_min,x_max, N)
y = stats.beta.pdf(randx, a, b)
randy = np.random.uniform(0,y.max(), N)
print(f'Integral value: {(x_max-x_min)*y.max()*(randy <= y).sum()/N}')

plt.figure(figsize=(10,6))
color = randy[:1000] <= y[:1000]
x = np.linspace(0, 1, 1000)
plt.plot(x, stats.beta.pdf(x, a, b))
plt.scatter(randx[:1000], randy[:1000], alpha=.2, c = color)
plt.xlabel('x')
plt.ylabel('density')
plt.show()