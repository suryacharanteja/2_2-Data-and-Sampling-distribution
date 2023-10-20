from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)

# Parameters for the population distribution
mu = 100  # Mean
sigma = 15  # Standard deviation

# Generate the population data
np.random.seed(42)
population_data = np.random.normal(mu, sigma, 100000)

# Sample size and number of samples
sample_size = 1
num_samples = 1000

# Initialize an array to store the sample means
sample_means = np.zeros(num_samples)

# Generate the sample means
for i in range(num_samples):
    sample = np.random.choice(population_data, sample_size)
    sample_means[i] = np.mean(sample)

# Plot the sample means
fig = plt.figure()
plt.hist(sample_means, bins=30, density=True, alpha=0.5, color='blue')

# Plot the normal distribution with the same mean and standard deviation
norm_x = np.linspace(mu - 4*sigma/np.sqrt(sample_size), mu + 4*sigma/np.sqrt(sample_size), 1000)
norm_y = 1/(sigma/np.sqrt(sample_size)*np.sqrt(2*np.pi)) * np.exp(-(norm_x - mu)**2/(2*(sigma/np.sqrt(sample_size))**2))
plt.plot(norm_x, norm_y, 'r-', lw=3)

plt.title('Central Limit Theorem')
plt.xlabel('Sample Means')
plt.ylabel('Density')

# Convert plot to PNG image and save to memory buffer
img_buffer = io.BytesIO()
fig.savefig(img_buffer, format='png')
img_buffer.seek(0)

# Define Flask route to serve the plot
@app.route('/')
def plot():
    return Response(img_buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
