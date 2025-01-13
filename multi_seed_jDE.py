import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Lists to store energy, nfes, and duration
energies = []
runtimes = []
nfes_list = []

# Run the program for 50 seeds
for seed in range(1, 51):
    result = subprocess.run([f'.\\Naloga_02.exe', "ABBBBBBABBBAB", "-seed", str(seed), "-target", "-5.6104", 
                             "-nfesLmt", "1000000", "-Np", "300", "-runtimeLmt", "3600"], capture_output=True, text=True)
    
    # Extract the values by splitting the output
    output_lines = result.stdout.split('\n')
    if len(output_lines) >= 3:
        energy = float(output_lines[0])
        nfes = int(output_lines[1])
        duration = float(output_lines[2])
        
        # Append values to lists
        energies.append(energy)
        nfes_list.append(nfes)
        runtimes.append(duration)
    else:
        print(f"Unexpected output format for seed {seed}")

# Calculate statistics
best_energy = min(energies)
worst_energy = max(energies)
average_energy = np.mean(energies)
std_dev_energy = np.std(energies)
average_runtime = np.mean(runtimes)
average_nfes = np.mean(nfes_list)

# Output results
print(f"Best energy: {best_energy}")
print(f"Worst energy: {worst_energy}")
print(f"Average energy: {average_energy}")
print(f"Standard deviation of energy: {std_dev_energy}")
print(f"Average runtime: {average_runtime} ms")
print(f"Average nfes: {average_nfes}")

# Plot all graphs in subplots
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(energies)
plt.ylabel('Energies')

plt.subplot(3, 1, 2)
plt.plot(runtimes)
plt.ylabel('Runtimes')

plt.subplot(3, 1, 3)
plt.plot(nfes_list)
plt.ylabel('NFES')

plt.tight_layout()
plt.show()