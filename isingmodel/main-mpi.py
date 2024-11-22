from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(rank, size)

# Start timer
comm.Barrier()  # Synchronize all processes before starting the timer
start_time = time.time()

# Lattice and model parameters
L = 128               # Size of the lattice along one dimension
T = 1                # Temperature
J = 1.0              # Coupling constant
steps = 128            # Number of Monte Carlo steps
k_B = 1              # Boltzmann constant
np.random.seed(42 + rank)  # Seed per rank for randomness (different for each process)

# Divide the lattice among processes
local_L = L // size  # Each process gets a section of the lattice
if rank == 0:
    lattice = np.random.choice([-1, 1], (L, L))  # Initialize lattice on master
else:
    lattice = None  # Other processes will receive their portion

# Scatter lattice portions to each process
local_lattice = np.zeros((local_L, L), dtype=int)
comm.Scatter(lattice, local_lattice, root=0)

# Function to perform a Monte Carlo update step
def monte_carlo_step(local_lattice, T, J, k_B):
    """Perform a Monte Carlo update step on the lattice portion."""
    for i in range(local_L):
        for j in range(L):
            S = local_lattice[i, j]
            # Calculate energy change
            neighbors_sum = (
                local_lattice[i, (j + 1) % L] + local_lattice[i, (j - 1) % L] +
                local_lattice[(i + 1) % local_L, j] + local_lattice[(i - 1) % local_L, j]
            )
            delta_E = 2 * J * S * neighbors_sum
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
                local_lattice[i, j] *= -1  # Flip spin

# Function to exchange borders with neighboring processes
def exchange_borders(local_lattice):
    """Exchange borders with neighboring processes."""
    top_border = local_lattice[0, :].copy()
    bottom_border = local_lattice[-1, :].copy()

    # Send bottom row to the process below, receive top row from above
    if rank < size - 1:
        comm.Sendrecv(bottom_border, dest=rank + 1, sendtag=0,
                      recvbuf=local_lattice[-1, :], source=rank + 1, recvtag=1)

    # Send top row to the process above, receive bottom row from below
    if rank > 0:
        comm.Sendrecv(top_border, dest=rank - 1, sendtag=1,
                      recvbuf=local_lattice[0, :], source=rank - 1, recvtag=0)

# Monte Carlo simulation
for step in range(steps):
    monte_carlo_step(local_lattice, T, J, k_B)
    exchange_borders(local_lattice)  # Sync border rows with neighbors

# Gather results back to the master process
gathered_lattice = None
if rank == 0:
    gathered_lattice = np.zeros((L, L), dtype=int)
comm.Gather(local_lattice, gathered_lattice, root=0)

# End timer
comm.Barrier()  # Synchronize all processes before stopping the timer
end_time = time.time()

# Print the time taken by the master process
if rank == 0:
    plt.imshow(gathered_lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)
    plt.title('Final Lattice Configuration')
    plt.show()
    print(f"Time taken for simulation: {end_time - start_time:.2f} seconds")
