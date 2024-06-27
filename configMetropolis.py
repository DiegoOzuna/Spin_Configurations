# The purpose of this file is to build off the spin configuration functions. We gradually update our lattices and record them 
# after a "large gap" of monte carlo iterations to ensure independence in spin configurations.

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import spinConfigs # will be holding the lattice functions...


def storeData(magnetTempData, fileName):
    df = pd.DataFrame(magnetTempData, columns=['Temperature', 'Magnetization'])
    df.to_csv(fileName, index=False)

########################################################################################################################
# purpose: montecarlo will simply generate a list of magnetizations after applying a metropolis algorithm where we  
#          repeatedly update the configuration of an inital lattice with spin ups and downs (+1,-1). 
#
# Params:
# lattice; the initial configuration of a 2d lattice (nxn)
# MCS; monte carlo steps that are set
# Temp; the max temperature of our system (will start from 1 and go to Temp in 0.5 increments)
# step; the MCS step we want to consistently keep measurements on (ex: every 1000th step we measure our system)
########################################################################################################################
def montecarlo(lattice, MCS, maxTemp, step):
    temp_steps = np.arange(1,maxTemp,0.5)
    latticeConfigurations = [0] * temp_steps.size                 # each index corresponds to a temperature (spin1 corresponds to temp 1)
    Elist = [0] * temp_steps.size                                 # each index corresponds to a temperature (spin1 corresponds to temp 1)
    
    Mlist = {t: [0] * (MCS//step) for t in temp_steps}  # we know at each MCS we save configurations at different temperatures (ie we know we need # of unique
                                                        # temp allocated lists of size MCS divided by the amount of steps before a measurement occurs)
    

    index = 0       #This index is used in order to store our data correctly within the Mlist every 1000 steps
    
    n = lattice.n

    for z in tqdm(range(MCS)):
        # do singular spin flips first
        for t in range(temp_steps.size):
            latticeConfigurations[t] = single_spin_flips(lattice, temp_steps[t])      # store final lattice from single_spin_flip for specific temperature

        # Calculate total energy of systems
        for t in range(temp_steps.size):
            Elist[t] = calculateTotalEnergy(latticeConfigurations[t])

        #Now we swap spins (parallel tempering)
        for t in range((temp_steps.size - 1)): #-1 to avoid getting out of bounds...
            temp = np.exp((Elist[t] - Elist[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]))

            # Generate some random, if this random is lower than the probability distribution...
            if (np.random.random() < temp):
                #Swap the spin configurations of this temperature with the next temperature
                # note: this list will be used for our measurements
                S_temp = latticeConfigurations[t]
                latticeConfigurations[t] = latticeConfigurations[t+1]
                latticeConfigurations[t+1] = S_temp

                #Ensure that we swap the energy of that spin configuration to its corresponding spot
                # note: this list will be used only for the scope of this for loop for calculations
                E_temp = Elist[t]
                Elist[t] = Elist[t+1]
                Elist[t+1] = E_temp

        #Measure the magnitudes of the lattices at that temperature
        if z % step == 0:
            for t in range(temp_steps.size):
                Mlist[temp_steps[t]][index] = latticeConfigurations[t].magnetization()
            index += 1
    
    return Mlist 



########################################################################################################################
# Purpose: The purpose of energy is to calculate the systems entire energy by summing up the lattice's spins together
#
# Params:
# lattice; the configuration of the 2d lattice (nxn)
########################################################################################################################   
def calculateTotalEnergy(lattice):
    L = int(np.sqrt(lattice.n))     # L^2 = N 
    E = 0
    for x in range(0, L):
        for y in range(0, L):
            E = E + lattice.config[x][y] * (lattice.config[(x+1)%L][y] + lattice.config[x][(y+1)%L])
    
    return E

    



########################################################################################################################
# Purpose: The purpose of single_spin_flips is in the name. We compute a check of the current observed spin wihtin the
#          lattice using the metropolis function and update that spin position if accepted. This will mimick the property 
#          if a single spin would flip within the lattice where at low temperatures it will be stubborn to change unless 
#          beneficial to the systems overall cost where as at higher temperatures, it will be more likely to accept changes 
#          that may or may not be beneficial.
#
# Params:
# lattice; the configuration of the 2d lattice (nxn)
# temp; the temperature of the system
########################################################################################################################
def single_spin_flips(lattice, temp):            #will apply single spin flips to the lattice using metropolis policy
    for i in range(lattice.n):
        for j in range(lattice.n):
            energy_of_flip = (2) * lattice.neighboring_cost(i,j)
                
            if(energy_of_flip <= 0):
                lattice.spin_flip(i,j)
            else:
                random = np.random.random() #generates some float between 0 and 1
                if random < np.exp(-energy_of_flip / temp):
                    lattice.spin_flip(i, j)

    new_lattice = spinConfigs.Lattice(lattice.n)
    new_lattice.config = np.copy(lattice.config)
    return new_lattice

########################################################################################################################
# Purpose: The purpose of equalibriated is to make sure that we have the system's lattice be closer to the probability
#          distribution (in theory exp(-(E(alpha)/kT)))
#
# Params:
# lattice; the initial configuration of the 2d lattice (nxn)
# temp; the temperature of the system
########################################################################################################################            
def equalibriated(lattice, temp):      #will assume that we could do 10000 steps to achieve an equalibriated state...
    for z in range(10000):
        for i in range(lattice.n):
            for j in range(lattice.n):
                energy_of_flip = (2) * lattice.neighboring_cost(i,j)
                
                if(energy_of_flip <= 0):
                    lattice.spin_flip(i,j)
                else:
                    random = np.random.random() #generates some float between 0 and 1
                    if random < np.exp(-energy_of_flip / temp):
                        lattice.spin_flip(i, j)
    
    new_lattice = spinConfigs.Lattice(lattice.n)
    new_lattice.config = np.copy(lattice.config)
    return new_lattice




lattice = spinConfigs.Lattice(4) #4x4 lattice
maxTemp = 5
MonteCarloSteps = 10**5
measureEvery = 1000

array = montecarlo(lattice, MonteCarloSteps, maxTemp, measureEvery)

print(array)
