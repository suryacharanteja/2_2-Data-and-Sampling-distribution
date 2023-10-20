import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

app = Flask(__name__)

# Parameters for the population distribution
mu = 100  # Mean
sigma = 15  # Standard deviation

# Generate the population data
np.random.seed(42)
population_data = np.random.normal(mu, sigma, 100000)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default sample size and number of samples
    sample_size = 1
    num_samples = 1000
    
    if request.method == 'POST':
        # Update sample size based on user input
        sample_size = int(request.form['sample_size'])
        
    # Initialize an array to store the sample means
    sample_means = np.zeros(num_samples)

    # Generate the sample means
    for i in range(num_samples):
        sample = np.random.choice(population_data, sample_size)
        sample_means[i] = np.mean(sample)

    # Plot the sample means
    plt.hist(sample_means, bins=30, density=True, alpha=0.5, color='blue')

    # Plot the normal distribution with the same mean and standard deviation
    norm_x = np.linspace(mu - 4*sigma/np.sqrt(sample_size), mu + 4*sigma/np.sqrt(sample_size), 1000)
    norm_y = 1/(sigma/np.sqrt(sample_size)*np.sqrt(2*np.pi)) * np.exp(-(norm_x - mu)**2/(2*(sigma/np.sqrt(sample_size))**2))
    plt.plot(norm_x, norm_y, 'r-', lw=3)

    plt.title('Central Limit Theorem')
    plt.xlabel('Sample Means')
    plt.ylabel('Density')
    plt.tight_layout()
    
    # Convert the plot to a base64-encoded image to display in the HTML
    import io
    from base64 import b64encode
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    img_data = b64encode(buf.getvalue()).decode()
    
    return render_template('index.html', img_data=img_data, sample_size=sample_size)

if __name__ == '__main__':
    app.run()
    