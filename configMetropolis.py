# The purpose of this file is to build off the spin configuration functions. We gradually update our lattices and record them 
# after a "large gap" of monte carlo iterations to ensure independence in spin configurations.

import numpy as np
import spinConfigs # will be holding the lattice functions...



#
# Below is an example being constructed, which was used to generalize into functions that are above...
# initialize 4x4 lattice...

Temp = [1,2,3,4,5]

MCS = 10**2     #monte carlo steps

for T in Temp:
    lattice = spinConfigs.initializeLattice(4)

    
    print("Current Temperature: ", T)
    print()
    print("original: \n", lattice)
    print()
    for z in range(MCS):
        for i in range(4):
            for j in range(4):
                energy_of_flip = -2 * spinConfigs.neighboringCost(lattice, i, j)
                
                if(energy_of_flip < 0):
                    spinConfigs.spinFlip(lattice, i, j)
                    #print("FLIPPED i, j = ", i , j)
                else:
                    random = np.random.random() #generates some float between 0 and 1
                    if random > np.exp(-energy_of_flip/T):
                        spinConfigs.spinFlip(lattice, i, j)

        if z % 25 == 0:
            print(lattice)
