import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pareto

# Generate Pareto distributed data
np.random.seed(42)  # For reproducibility
alpha = 2.5  # Shape parameter
xm = 500_000  # Scale parameter (minimum value)
size = 10_000_000  # Sample size

data = pareto.rvs(alpha, scale=xm, size=size)


# Define a function to calculate mean excess
def mean_excess(data, threshold):
    excesses = data[data > threshold] - threshold
    return excesses.mean() if len(excesses) > 0 else np.nan


# Calculate mean excesses for different thresholds
thresholds = np.linspace(np.min(data), np.max(data), 100)
mean_excesses = [mean_excess(data, u) for u in thresholds]

# Create a DataFrame for plotting
df_mean_excess = pd.DataFrame({"threshold": thresholds, "mean_excess": mean_excesses})

# Plot mean excess plot using Plotly Express
fig = px.scatter(
    df_mean_excess,
    x="threshold",
    y="mean_excess",
)
fig.show()
