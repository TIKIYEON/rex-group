import random
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


#For this question we choose the proposal distribution q(x) to be a uniform distribution on the interval
# [0 : 15]. You can generate samples from such a distribution using for instance Python’s random package.

#You have to write a python program that produces a set of resampled robot poses x using the Sampling 
#Importance-Resampling algorithm and the above stated pose distribution p(x) and proposal distribution 
# q(x). Show the distribution of samples after the resampling step for k = 20, 100, 1000 samples /
#particles. Plot a histogram of the samples together with the wanted pose distribution p(x) (Hint: Take
#care the the histogram should be scaled as a probability density function to be comparable with p(x)).
#How well does the histogram of samples fit with p(x) for the different choices of k? Can you imagine
#any problems occurring when using a uniform proposal distribution with our particular choice of p(x)?

def q_x(k):
    q = np.random.uniform(0, 15, k)
    return q

#p(x) = 0.3 · N(x; 2.0, 1.0) + 0.4 · N(x; 5.0, 2.0) + 0.3 · N(x; 9.0, 1.0) 
#a function to define the target distribution for 
def p_x(x):
    pdf = (0.3 * norm.pdf(x, 2.0, 1.0) + 0.4 * norm.pdf(x, 5.0, 2.0) + 0.3 * norm.pdf(x, 9.0, 1.0))
    return pdf

def resample(particles, weights):
    #normalize the weights by dividing each particle's weight over the sum of weights
    weights = weights / sum(weights)
    #resample the particles based on their weights
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]
    
def SIR(k):
    #generate k random samples using q(x)
    particles = q_x(k)
    #print(particles)
    
    #assign the weights using p(x)
    weights = p_x(particles)
    
    #resample the particles based on their weight
    resampled_particles = resample(particles, weights)
    
    return resampled_particles

#function to plot the particles after being resampled and the expected/target distribution of p(x)
def plot_histograms(k_values):
    x_range = np.linspace(0, 15, 1000)
    
    # Plot the target distribution p(x) for reference
    plt.figure(figsize=(10, 8))
    
    for i, k in enumerate(k_values):
        resampled_particles = SIR(k)
        
        # Plot the histogram of resampled particles
        plt.subplot(len(k_values), 1, i+1)
        plt.hist(resampled_particles, bins=30, density=True, alpha=0.6, label=f'Resampled (k={k})')
        
        # Overlay the target distribution p(x) for comparison
        plt.plot(x_range, p_x(x_range), label='Target distribution p(x)', color='red')
        plt.title(f'Resampling with k={k}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

k_values = [20, 100, 1000]
plot_histograms(k_values)