#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.

import numpy as np

#This function will create our initial lattice space.
def initializeLattice(n):
    lattice = np.zeros((n,n))
    fillLattice(lattice, n)
    return(lattice)

#This function is meant to fill in our initial lattice space with "spins" (up/down)
def fillLattice(lattice, n):
    for x in range(n):
        for y in range(n):
            lattice[x][y] = np.random.choice([1, -1])


# below was inital lattice example, to then generalize above...
# lattice = np.zeros((32,32))

# print("Initial Empty Lattice Space")
# print(lattice)

# for x in range(32):
#         for y in range(32):
#             lattice[x][y] = np.random.choice([1, -1])

# print("Initial Lattice Space")
# print(lattice)