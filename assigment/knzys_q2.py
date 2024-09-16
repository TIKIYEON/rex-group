import random
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Function q(x) will take the number of samples and return a random list q generated using Normal Distribution
# where the mean is 5 and the standard deviation is 2.
def q_x(k):  
    q = np.random.normal(5.0, 2.0, k)
    return q

# A function p(x) to define the target distribution where  
# p(x) = 0.3 · N(x; 2.0, 1.0) + 0.4 · N(x; 5.0, 2.0) + 0.3 · N(x; 9.0, 1.0) 
def p_x(x):
    pdf = (0.3 * norm.pdf(x, 2.0, 1.0) + 0.4 * norm.pdf(x, 5.0, 2.0) + 0.3 * norm.pdf(x, 9.0, 1.0))
    return pdf


# We will use this function to normalize the weights of each probability by dividing each particle's weight 
# over the sum of weights such that the values of weights array add up to 1. Resampling is done such that the
# particles with higher weights are more likely to be picked multiple times. The random choice function does 
# that by picking random indices from the list of particles based on their weights. Thus, particles with higher 
# weights are more likely to have their indices picked more often. Then using the newly generated list of indices
# we return the list of particles such that the particles with higher weighst appear more often in the list 
# of particles.
def resample(particles, weights):
    weights = weights / sum(weights)
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# Now putting all these functions together in the SIR, Sampling Importance-Resampling Algorithm, function. We will 
# q(x) to generate random values for the samples/particles representing the robot's position, random x. then we
# use the probability distribution function p(x) to determine the weights/importance of these random particles. then 
# based on these weights, we will resample the particles to kow which ones are more likely to model the robot's position.
def SIR(k):
    # proposal sampling
    particles = q_x(k)

    # importance Sampling
    # unlike when we were using uniform sampling for the sampling function, we are now using normal distribution for sampling
    # so now we are generating values that might be likely according to q(x) but not p(x), so now the weight of each particle
    # is calculated as the ratio of how likely/important it is according to p(x) compared to q(x)    
    weights = p_x(particles)/norm.pdf(particles,5.0,2.0)
    
    # resampling
    resampled_particles = resample(particles, weights)
    
    return resampled_particles

# Function to plot the resampled particles and the expected/target distribution using p(x).
def plot_histograms(k_values):
    
    # First off, we need to generate a range of x values for p(x) over the range from [0:15] as p(x) is a continuous 
    # function so we need to evaluate it over a range of x values to compare it to the SIR algorithm. linspace function
    # generates values between 0 and 15 such that it returns a range of 1000 values in the range [0:15].
    x_range = np.linspace(0, 15, 1000)
    
    plt.figure(figsize=(10, 8))
    
    for i, k in enumerate(k_values):
        # Get the resampled particles list from SIR
        resampled_particles = SIR(k)
        
        # This line is responsible for dividing our page into equal parts, and since we have 20, 100, and 1000, it will
        # divide our screen into 3 parts for the 3 iterations/graphs.
        plt.subplot(len(k_values), 1, i+1)

        # Plot the histogram of resampled particles        
        plt.hist(resampled_particles, bins=30, density=True, alpha=0.6, label=f'Resampled (k={k})')
        
        # Overlay the target distribution p(x) for comparison by sending the x_range list to p(x) and plotting it.
        plt.plot(x_range, p_x(x_range), label='Target distribution p(x)', color='red')
        
        # Adding labels and title
        plt.xlabel('Robot Position (x)') 
        plt.ylabel('Probability Density')  
        plt.title(f'Resampling with k={k}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

k_values = [20, 100, 1000]
plot_histograms(k_values)
