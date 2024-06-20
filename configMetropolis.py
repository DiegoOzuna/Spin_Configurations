# The purpose of this file is to build off the spin configuration functions. We gradually update our lattices and record them 
# after a "large gap" of monte carlo iterations to ensure independence in spin configurations.

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool # this is to let us do the calculations of each temp seperate
import spinConfigs # will be holding the lattice functions...


def storeData(magnetTempData):
    df = pd.DataFrame(magnetTempData, columns=['Temperature', 'Magnetization'])
    df.to_csv("magnetizations.csv", index=False)


def montecarlo(lattice, MCS, Temp, equilibration_steps, step):
    M = []
    n = lattice.n
    for z in tqdm(range(MCS + equilibration_steps)):
        for i in range(n):
            for j in range(n):
                a = np.random.choice(range(n))
                b = np.random.choice(range(n))
                energy_of_flip = np.multiply(-2, lattice.neighboring_cost(a,b))
                
                if(energy_of_flip < 0):
                    lattice.spin_flip(a,b)
                else:
                    random = np.random.random() #generates some float between 0 and 1
                    if random < np.exp(np.divide((-energy_of_flip),Temp)):
                        lattice.spin_flip(a, b)

        if z >= equilibration_steps and z % step == 0:
            M.append(lattice.magnetization())

    return M



def compute_magnetization(temp):
    lattice = spinConfigs.Lattice(16)
    magnetization = montecarlo(lattice, 10**6, temp, 100000, 1000)
    return (temp, magnetization)



##########################################################################################

if __name__ == '__main__':
    temps = np.arange(1, 5, 0.25)
    with Pool() as p:
        with tqdm(total=len(temps)) as pbar:
            magnetTempData = []
            for i, result in tqdm(enumerate(p.imap_unordered(compute_magnetization, temps))):
                magnetTempData.append(result)
                pbar.update()
    storeData(magnetTempData)


#
# Below is an example being constructed, which was used to generalize into functions that are above...
# initialize 4x4 lattice...

# Temp = [1,2,3,4,5]

# MCS = 10**2     #monte carlo steps

# for T in Temp:
#     lattice = spinConfigs.Lattice(16)
#     lattice.

    
#     print("Current Temperature: ", T)
#     print()
#     print("original: \n", lattice)
#     print()
#     for z in range(MCS):
#         for i in range(4):
#             for j in range(4):
#                 energy_of_flip = -2 * lattice.neighboring_cost(i,j)
                
#                 if(energy_of_flip < 0):
#                     lattice.spin_flip(i,j)
#                     #print("FLIPPED i, j = ", i , j)
#                 else:
#                     random = np.random.random() #generates some float between 0 and 1
#                     if random < np.exp(-energy_of_flip/T):
#                         lattice.spin_flip(i, j)

#         if z % 25 == 0:
#             print(lattice.config)

