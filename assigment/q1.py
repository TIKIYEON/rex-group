import random
import scipy as sp


def p(x):
    return 0.3+sp.stats.norm.pdf(x, 2.0, 1.0)
    + 0.4+sp.stats.norm.pdf(x, 5.0, 2.0)
    + 0.3+sp.stats.norm.pdf(x, 9.0, 1.0)

def sir(k):
    samples = [random.uniform(0, 15) for _ in range(k)]

    weights = [p(x) for x in samples]
    sumWeights = sum(weights)
    normWeights = [w / sumWeights for w in weights]

    reSampling = random.choices(samples, weights=normWeights, k=k)

    return reSampling

k20 = sir(20)
k100 = sir(100)
k1000 = sir(1000)
print(k20)
