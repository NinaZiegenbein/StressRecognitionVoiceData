import glob
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Find the CSV file using glob
csv_files = glob.glob('/data/dst_tsst_22_bi_multi_nt_lab/raw/dstvtsst_limesurvey_v5_14-03-23.csv')

# Read and modify the CSV file
data = pd.read_csv(csv_files[0])  # Convert CSV data to pd

# Make changes to the CSV data (example: multiply all values by 2)
modified_data = data.drop(columns=["lastpage"])

# Save the modified data as a new CSV file
modified_data.to_csv("/homes/nziegenbein/test_modified.csv", index=False)

# Create a simple Matplotlib plot
# Generate example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example Plot')

# Save the Matplotlib figure
plt.savefig("/homes/nziegenbein/test_plot.png")
plt.close()  # Close the plot for the next iteration

print(f"Modified file saved as!")
print(f"Plot saved as!")
