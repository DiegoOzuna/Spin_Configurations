# The purpose of this file is to build off the spin configuration functions. We gradually update our lattices and record them 
# after a "large gap" of monte carlo iterations to ensure independence in spin configurations.

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import spinConfigs # will be holding the lattice functions...


def storeData(dictionaries, filenames):
    for dictionary, filename in zip(dictionaries, filenames):
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(list(dictionary.items()), columns=['Temperature', 'Measurement'])

        # Convert the lists in the 'Measurement' column to string
        df['Measurement'] = df['Measurement'].apply(lambda x: str(x))

        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)


def montecarlo(MCS, maxTemp, step):
    """
    purpose: montecarlo will simply generate a list of magnetizations after applying a metropolis algorithm where we  
             repeatedly update the configuration of an inital lattice with spin ups and downs (+1,-1). 
    
    Params:
    
    MCS; monte carlo steps that are set
    
    maxTemp; the max temperature of our system (will start from 1 and go to Temp in 0.5 increments)
    
    step; the MCS step we want to consistently keep measurements on (ex: every 1000th step we measure our system)
    """
    temp_steps = np.arange(0.5,maxTemp,0.5)
    
    # Generate one set of lattice configurations
    latticeConfigurations = [spinConfigs.Lattice(8) for _ in range(temp_steps.size)]

    # Generate replicas of the above configurations
    latticeConfigurations1 = [config.genReplica() for config in latticeConfigurations]  # S1
    latticeConfigurations2 = [config.genReplica() for config in latticeConfigurations]  # S2
    
    Elist1 = [0] * temp_steps.size # each index corresponds to a temperature (spin1 corresponds to temp 1)
    Elist2 = [0] * temp_steps.size
    
    Mlist1 = {t: [0] * (MCS//step) for t in temp_steps}  # we know at each MCS we save configurations at different temperatures (ie we know we need # of unique
    Mlist2 = {t: [0] * (MCS//step) for t in temp_steps}  # temp allocated lists of size MCS divided by the amount of steps before a measurement occurs)
    
    Qlist = {t: [0] * (MCS//step) for t in temp_steps}  # Overlap measurements
    

    index = 0       #This index is used in order to store our data correctly within the Mlists every 1000 steps
    
    n = latticeConfigurations1[0].n

    for z in tqdm(range(MCS)):
        # do singular spin flips first
        for t in range(temp_steps.size):
            latticeConfigurations1[t] = single_spin_flips(latticeConfigurations1[t], temp_steps[t])      # store final lattice from single_spin_flip for specific temperature
            latticeConfigurations2[t] = single_spin_flips(latticeConfigurations2[t], temp_steps[t])

        # Calculate total energy of systems
        for t in range(temp_steps.size):
            Elist1[t] = calculateTotalEnergy(latticeConfigurations1[t])
            Elist2[t] = calculateTotalEnergy(latticeConfigurations2[t])

        #Now we swap spins (parallel tempering)
        for t in range((temp_steps.size - 1)): #-1 to avoid getting out of bounds...
            temp1 = np.exp((Elist1[t] - Elist1[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]))
            temp2 = np.exp((Elist2[t] - Elist2[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]))

            # Generate some random, if this random is lower than the probability distribution...
            
            #For S1
            if (np.random.random() < temp1):
                #Swap the spin configurations of this temperature with the next temperature
                # note: this list will be used for our measurements
                S_temp = latticeConfigurations1[t]
                latticeConfigurations1[t] = latticeConfigurations1[t+1]
                latticeConfigurations1[t+1] = S_temp

                #Ensure that we swap the energy of that spin configuration to its corresponding spot
                # note: this list will be used only for the scope of this for loop for calculations
                E_temp = Elist1[t]
                Elist1[t] = Elist1[t+1]
                Elist1[t+1] = E_temp

            

            #For S2
            if (np.random.random() < temp2):
                S_temp = latticeConfigurations2[t]
                latticeConfigurations2[t] = latticeConfigurations2[t+1]
                latticeConfigurations2[t+1] = S_temp
                E_temp = Elist2[t]
                Elist2[t] = Elist2[t+1]
                Elist2[t+1] = E_temp

        #Measure the magnitudes of the lattices, measure the overlap between the two...
        if z % step == 0:
            for t in range(temp_steps.size):
                Mlist1[temp_steps[t]][index] = latticeConfigurations1[t].magnetization()
                Mlist2[temp_steps[t]][index] = latticeConfigurations1[t].magnetization()
                Qlist[temp_steps[t]][index] = calculate_overlap(latticeConfigurations1[t], latticeConfigurations2[t])
            index += 1
    
    return Mlist1, Mlist2, Qlist 

def calculate_overlap(lattice1, lattice2):
    """
    This function calculates the overlap between two spin configurations.
    The overlap is defined as the dot product of the spin vectors divided by the total number of spins.

    Parameters:
    lattice1 (Lattice): The first spin configuration.
    lattice2 (Lattice): The second spin configuration.

    Returns:
    float: The overlap between the two spin configurations.
    """
    return np.sum(lattice1.config * lattice2.config) / lattice1.config.size


  
def calculateTotalEnergy(lattice):
    """
    Purpose: The purpose of energy is to calculate the systems entire energy by summing up the lattice's spins together
    
    Params:
    lattice; the configuration of the lattice (nxnxn)
    """
    L = lattice.n     # L^3 = N ; therefore we only sum three sides
    E = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                E = E - lattice.config[x][y][z] * (lattice.config[(x+1)%L][y][z] + lattice.config[x][(y+1)%L][z] + lattice.config[x][y][(z+1)%L])
    
    return E

    



def single_spin_flips(lattice, temp):            #will apply single spin flips to the lattice using metropolis policy
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
    for i in range(lattice.n):
        for j in range(lattice.n):
            for z in range(lattice.n):
                energy_of_flip = (2) * lattice.neighboring_cost(i,j,z)
                    
                if(energy_of_flip <= 0):
                    lattice.spin_flip(i,j,z)
                else:
                    random = np.random.random() #generates some float between 0 and 1
                    if random < np.exp(-energy_of_flip / temp):
                        lattice.spin_flip(i, j, z)

    new_lattice = spinConfigs.Lattice(lattice.n)
    new_lattice.config = np.copy(lattice.config)
    return new_lattice




maxTemp = 2
MonteCarloSteps = 10**5
measureEvery = 1000

dictionaries = montecarlo(MonteCarloSteps, maxTemp, measureEvery)
filenames = ["Magnetization1_8x8.csv", "Magnetization2_8x8.csv", "Overlap_8x8.csv"]

storeData(dictionaries, filenames)







########################################################################################################################
# # Purpose: The purpose of equalibriated is to make sure that we have the system's lattice be closer to the probability
# #          distribution (in theory exp(-(E(alpha)/kT)))
# #
# # Params:
# # lattice; the initial configuration of the 2d lattice (nxn)
# # temp; the temperature of the system
# ########################################################################################################################            
# def equalibriated(lattice, temp):      #will assume that we could do 10000 steps to achieve an equalibriated state...
#     for z in range(10000):
#         for i in range(lattice.n):
#             for j in range(lattice.n):
#                 energy_of_flip = (2) * lattice.neighboring_cost(i,j)
                
#                 if(energy_of_flip <= 0):
#                     lattice.spin_flip(i,j)
#                 else:
#                     random = np.random.random() #generates some float between 0 and 1
#                     if random < np.exp(-energy_of_flip / temp):
#                         lattice.spin_flip(i, j)
    
#     new_lattice = spinConfigs.Lattice(lattice.n)
#     new_lattice.config = np.copy(lattice.config)
#     return new_lattice
