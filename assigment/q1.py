import random
import scipy as sp

qx = random.uniform(0, 15)


def p(x):
    0.3+sp.stats.uniform.pdf(x, 2.0, 1.0)
    + 0.4+sp.stats.uniform.pdf(x, 5.0, 2.0)
    + 0.3+sp.stats.uniform.pdf(x, 9.0, 1.0)

def sir(k):
    samples = [qx for i in range(k)]

    weights = [p(x) for x in samples]
    sumWeights = sum(weights)
    normWeights = [w / sumWeights for w in weights]

    reSampling = random.choices(samples, normWeights, k)

    return reSampling

