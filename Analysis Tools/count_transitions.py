import numpy as np
from scipy import sparse as sps
from os import makedirs
from os.path import join, exists
from matplotlib import pyplot as plt, ticker
from scipy.stats import norm
from scipy.optimize import fsolve
import math

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2

#Weights at different epochs, and log log plots
#Scaling influence of the architecture week

# Load the file
file_path = r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\HM_and_AI_models\VAE_Model\x_stoch.npy'
data = np.load(file_path)



print(data.shape)
print("Length of array is: ", len(data[:, :, 60]))


upper_bound = 53.8 / 2.8935
lower_bound = 1.75 / 2.8935


left_limit = 0
right_limit = 299400

#Grabbed the 61st element of the zonal wind
y_values = data[left_limit:right_limit, 1, 60]
x_values = np.arange(right_limit - left_limit)

"""Plot for the timeseries"""
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.xlabel('X Axis (Time steps)')
plt.axhline(y = upper_bound, color = 'b', linestyle = '--') 
plt.axhline(y = lower_bound, color = 'r', linestyle = '--') 
plt.ylabel('Y Axis (Zonal Wind @ 60)')
plt.title('Plot of zonal wind at index 60')


above_upper_bound_weak = [(index, y) for index, y in enumerate(y_values) if y > upper_bound]
below_lower_bound_weak = [(index, y) for index, y in enumerate(y_values) if y < lower_bound]

print("Above upper bound tuples:", len(above_upper_bound_weak))
print("Below lower bound tuples:", len(below_lower_bound_weak))


"""Print only the first n elements of the array"""
# upper_limit = 10000
# weak_state_ten = data[:upper_limit, 1, 50:]
# y_values_ten = np.mean(weak_state, axis=1)
# above_upper_bound_weak_ten = [(index, y) for index, y in enumerate(y_values_ten) if y > upper_bound]
# below_lower_bound_weak_ten = [(index, y) for index, y in enumerate(y_values_ten) if y < lower_bound]
"""Code Ends"""



"""Finding Transitions"""
transitions = []
above_upper = False  # Flag to track if we are above upper bound
time_to_transition = []
time_taken = 0
time_to_transition_two = []

transitions_two = []
below_lower = False  # Flag to track if we are below lower bound

time_taken = 0
for i in range(1, len(y_values)):
    time_taken += 1


    if y_values[i] > upper_bound:
        above_upper = True
        

    elif above_upper and y_values[i] < lower_bound:

        time_to_transition_two.append(time_taken)
        transitions.append(i)
        time_taken = 0
        above_upper = False  # Reset flag after transition


    if y_values[i] < lower_bound:
        below_lower = True

    elif below_lower and y_values[i] > upper_bound:
        transitions_two.append(i)


        time_to_transition.append(time_taken)
        time_taken = 0
        below_lower = False  # Reset flag after transition

print("Time taken list to transtion: " , time_to_transition)
print("Time taken list to transtion: " , time_to_transition_two)

print("Number of transitions from below lower to above upper bound:", len(transitions_two))
print("Number of transitions from above upper to below lower bound:", len(transitions))

"""Plot the Vertical Lines"""
for val in transitions:
    plt.axvline(x = val, color = 'r', label = 'axvline - full height')
for val in transitions_two:
    plt.axvline(x = val, color = 'b', label = 'axvline - full height')
plt.show()


""" Histogram code found here """



data = time_to_transition

bins = np.linspace(math.ceil(min(data)), 
                   math.floor(max(data)),
                   10) # fixed number of bins

plt.xlim([min(data)-5, max(data)+5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()


data = time_to_transition_two

bins = np.linspace(math.ceil(min(data)), 
                   math.floor(max(data)),
                   10) # fixed number of bins

plt.xlim([min(data)-5, max(data)+5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()