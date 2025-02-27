import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

# Load data
weights_file = r'C:\Users\danie\OneDrive\Desktop\Holton Mass Model\SHORT\model_weights.weights.h5.npz'
x_file = r'C:\Users\danie\OneDrive\Desktop\Holton Mass Model\SHORT\150klongtimeseries\x_stoch.npy'
data = np.load(weights_file)
data_real = np.load(x_file)

# Extract predictions and real data
predictions = data['predictions'].squeeze()  # Shape becomes (100000, 75)
real_data = data_real[:, 0, 63]  # Extract real zonal wind data at index 60

# Define bounds
upper_bound = 53.8 / 2.8935
lower_bound = 1.75 / 2.8935

# Function to calculate transition durations
def calculate_transition_durations(y_values, upper_bound, lower_bound):
    times_between_transitions = []
    transition_start = None
    above_upper = False
    below_lower = False

    for i in range(1, len(y_values)):
        if y_values[i] < lower_bound:  
            below_lower = True
            above_upper = False
        elif y_values[i] > upper_bound:  
            if below_lower and transition_start is not None:
                times_between_transitions.append(i - transition_start)
                transition_start = None  
            above_upper = True
            below_lower = False

        if below_lower and transition_start is None:
            transition_start = i

    return times_between_transitions

# Compute transition durations for real data
real_durations = calculate_transition_durations(real_data, upper_bound, lower_bound)

# Compute CCDF
real_data_sorted = np.sort(real_durations)
ccdf_real = 1 - np.arange(1, len(real_data_sorted) + 1) / len(real_data_sorted)

# Filter valid data (exclude zero or negative CCDF values)
valid_indices = ccdf_real > 0  # Avoid log(0) issues
x_fit = real_data_sorted[valid_indices]  # Keep x in linear scale
y_fit = np.log(ccdf_real[valid_indices])  # Apply log transformation to y

# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)

# Convert back to exponential form (y = e^(slope*x + intercept))
x_line = np.linspace(min(x_fit), max(x_fit), 100)
y_line = np.exp(slope * x_line + intercept)  # Convert back from log scale

# Plot CCDF and best-fit line
plt.figure(figsize=(10, 6))
plt.step(real_data_sorted, ccdf_real, where='post', label='Real Data CCDF', linewidth=2, linestyle='--')
plt.plot(x_line, y_line, 'r-', label=f'Exponential Fit (slope={slope:.4f})', linewidth=2)

plt.xlabel('Time Duration (Steps)')
plt.ylabel('CCDF')
plt.title('CCDF of Time Between B->A and A->B Transitions (Exponential Fit)')
plt.yscale("log")  # Keep y-axis in log scale
plt.xscale("linear")  # Keep x-axis in linear scale
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
