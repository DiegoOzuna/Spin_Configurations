# worker.py
import numpy as np
from numpy import random
from tqdm import tqdm
from numba import jit

####THE WORKER FILE IS A COMBINATION OF BOTH SPINCONFIGS.py AND CONFIGMETROPOLIS.py#####

@jit(nopython=True)
def Lattice(n):  #This function will create our initial 3D lattice space. given bonds and lattice is optional
    return np.random.choice(np.array([1, -1]), size=(n, n, n))


@jit(nopython=True)
def neighboring_cost(lattice, bonds, n, i, j, z):    #This function is meant to calculate the local cost of S_ijz
    currentNode = lattice[i][j][z]
    #The local cost is the summation of all the neighboring spins * their weight with respect to the current node
    #multiplied by the current node...
    # Calculate indices for neighbors

    # This is within the same lattice space... (2d)
    above = (i, (j+1)%n, z)
    below = (i, (j-1)%n, z)
    right = ((i+1)%n, j, z)
    left = ((i-1)%n, j, z)

    # This is entering the other lattice space... (3d)
    forward = (i, j, (z+1)%n)
    backward = (i, j, (z-1)%n)

        # Calculate bond values for neighbors
    neighbors = [above, below, right, left, forward, backward]
    bond_values = [lattice[n]* bonds[n] for n in neighbors]

    return currentNode * sum(bond_values)

@jit(nopython=True)
def spin_flip(lattice, i, j, z):          #This function is meant to just flip the spin of S_ijz passed...
    lattice[i][j][z] *= -1

@jit(nopython=True)
def magnetization(lattice, bonds, n):            #This function will grab the magnetization of the 3d lattice
    return np.sum(np.multiply(lattice, bonds))/ (n ** 3)
    

@jit(nopython=True)
def Bonds(n):  #This function will create our initial bond.
    return np.random.choice(np.array([1,-1]), size=(n, n, n)) # Generate bonds 
    


@jit(nopython=True)
def calculateTotalEnergy(n, lattice, bonds):
    """
    Purpose: The purpose of energy is to calculate the systems entire energy by summing up the lattice's spins together
    
    Params:
    lattice; the configuration of the lattice (nxnxn)
    """
    total_energy = 0
    nx, ny, nz = lattice.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate the energy contribution from each spin and its neighbors
                E = -lattice[i, j, k] * (
                    lattice[(i+1)%nx, j, k] * bonds[(i+1)%nx, j, k] +
                    lattice[i, (j+1)%ny, k] * bonds[i, (j+1)%ny, k] +
                    lattice[i, j, (k+1)%nz] * bonds[i, j, (k+1)%nz]
                )
                # Add the energy contribution to the total energy
                total_energy += E

    return total_energy


@jit(nopython=True)
def single_spin_flips(lattice, bonds, n, temp):            #will apply single spin flips to the lattice using metropolis policy
    '''
    Purpose: The purpose of single_spin_flips is in the name. We compute a check of the current observed spin within the
    lattice using the metropolis function and update that spin position if accepted. This will mimick the property 
    if a single spin would flip within the lattice where at low temperatures it will be stubborn to change unless 
    beneficial to the systems overall cost where as at higher temperatures, it will be more likely to accept changes 
    that may or may not be beneficial.

    Params:
    
    lattice; the configuration of the 3d lattice (nxnxn)
    
    temp; the temperature of the system
    '''
    random_numbers = random.random(n**3)  # generates some float between 0 and 1
    for i in range(n):
        for j in range(n):
            for z in range(n):
                idx = i * n**2 + j * n + z  # calculate 1D index from 3D indices
                energy_of_flip = 2 * neighboring_cost(lattice, bonds, n, i, j, z)
                if random_numbers[idx] < np.exp(-energy_of_flip / temp):
                    spin_flip(lattice, i, j, z)
    
    return lattice



def montecarlo(L, MCS, temp_steps, step, equilock):

    # Initialize an empty dictionary
    lattice1_dict = {}
    lattice2_dict = {}


    """
    purpose: the purpose of this function is to apply single spin flips and parallel tempering to two lists of lattices 
    using a given bond configuration shared between them. These lists are of the size of 0.8 to maxTemp increments long.
    After the process of flipping is finished the lattices will be stored every 1000th MCS step across the varying temperatures. 
    
    Params:

    L; the lattice size
    
    MCS; monte carlo steps that are set for each "realization"
    
    temp_steps; the temperatures that our system undergoes
    
    step; the MCS step we want to consistently keep measurements on (ex: every 1000th step we measure our system)

    equilock; the amount of MCS steps we ignore in measuring to ensure our system is equilibrated
    """
    

    #Define an initial bond configuration that will be used throughout the sim...
    bonds = Bonds(L) #will make a random config but only store the bonds of this random config
    

    # Generate replicas of the above configurations
    latticeConfigurations1 = np.array([Lattice(L) for _ in range(temp_steps.size)])  # S1
    latticeConfigurations2 = np.array([Lattice(L) for _ in range(temp_steps.size)])  # S2

    Elist1 = np.zeros(temp_steps.size) # each index corresponds to a temperature (spin1 corresponds to temp 1)
    Elist2 = np.zeros(temp_steps.size)

    # Mlist1 = np.zeros((temp_steps.size, MCS//step))  # we know at each MCS we save configurations at different temperatures (ie we know we need # of unique
    # Mlist2 = np.zeros((temp_steps.size, MCS//step))  # temp allocated lists of size MCS divided by the amount of steps before a measurement occurs)

    # Qlist = np.zeros((temp_steps.size, MCS//step))  # Overlap measurements individually


    for z in tqdm(range(MCS)):
        # do singular spin flips first
        for t in range(temp_steps.size):
            single_spin_flips(latticeConfigurations1[t], bonds, L, temp_steps[t])      # store final lattice from single_spin_flip for specific temperature
        for t in range(temp_steps.size):
            single_spin_flips(latticeConfigurations2[t], bonds, L, temp_steps[t])

        # Calculate total energy of systems
        for t in range(temp_steps.size):
            Elist1[t] = calculateTotalEnergy(L, latticeConfigurations1[t], bonds)
        for t in range(temp_steps.size):
            Elist2[t] = calculateTotalEnergy(L, latticeConfigurations2[t], bonds)

        #Now we swap spins (parallel tempering)
        for t in range((temp_steps.size - 1)): #-1 to avoid getting out of bounds...
            temp1 = np.exp((Elist1[t] - Elist1[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]))
            temp2 = np.exp((Elist2[t] - Elist2[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]))

            # Generate some random, if this random is lower than the probability distribution...
            
            #For S1
            if (random.random() < temp1):
                #Swap the spin configurations of this temperature with the next temperature
                latticeConfigurations1[t], latticeConfigurations1[t+1] = latticeConfigurations1[t+1], latticeConfigurations1[t]

                #Ensure that we swap the energy of that spin configuration to its corresponding spot
                Elist1[t], Elist1[t+1] = Elist1[t+1], Elist1[t]

            #For S2
            if (random.random() < temp2):
                latticeConfigurations2[t], latticeConfigurations2[t+1] = latticeConfigurations2[t+1], latticeConfigurations2[t]
                Elist2[t], Elist2[t+1] = Elist2[t+1], Elist2[t]


        # Store the lattices for the temperatures respectively...
        if z % step == 0 and z > equilock:
            for t in range(temp_steps.size):
                # If the temperature is not in the dictionary, add it with an empty list as the value
                if temp_steps[t] not in lattice1_dict:
                    lattice1_dict[temp_steps[t]] = []
                    lattice2_dict[temp_steps[t]] = []
                
                # Append the 3D lattice to the list for this temperature
                lattice1_dict[temp_steps[t]].append(latticeConfigurations1[t])
                lattice2_dict[temp_steps[t]].append(latticeConfigurations2[t])
            
    return lattice1_dict, lattice2_dict

def worker(params):
    L, MonteCarloSteps, temp_steps, measureEvery = params
    spins1, spins2 = montecarlo(L, MonteCarloSteps, temp_steps, measureEvery, MonteCarloSteps/2)
    return spins1, spins2


# #Testing lattice creation.....
# print("lattice creation")
# n = 3
# lattice = Lattice(n)
# print(lattice)

# print("\n\n")

# #Testing spin flip....
# print("spin flip...")
# spin_flip(lattice, 0,0,0)
# print(lattice)

# print("\n\n")

# #Testing bond creation....
# print("bond creation")
# bond = Bonds(n)
# print(bond)

# print("\n\n")

# #Testing neighboring cost....
# print("calc cost at 0,0,0")
# print(neighboring_cost(lattice, bond, n, 0,0,0))

# print("\n\n")