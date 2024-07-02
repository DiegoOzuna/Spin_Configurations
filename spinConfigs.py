#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.
#It is also meant to handle operations done on the lattice itself.

import numpy as np

class Lattice:
    def __init__(self, n):      #This function will create our initial 3D lattice space.
        self.n = n
        self.config = self.fill_lattice()

    def fill_lattice(self):     #This function is meant to fill in our initial lattice space with "spins" (up/down)
        config = np.random.choice([1, -1], size=(self.n, self.n, self.n))
        return config


    def neighboring_cost(self, i, j, z):       #This function is meant to calculate the local cost of S_ijz
        currentNode = self.config[i][j][z]

        # This is within the same lattice space... (2d)
        aboveNode = self.config[i][(j+1)%self.n][z]
        belowNode = self.config[i][(j-1)%self.n][z]
        rightNode = self.config[(i+1)%self.n][j][z]
        leftNode = self.config[(i-1)%self.n][j][z]

        # This is entering the other lattice space... (3d)
        forwardNode = self.config[i][j][(z+1)%self.n]
        backwardNode = self.config[i][j][(z-1)%self.n]

        return 1 * currentNode * (aboveNode+belowNode+rightNode+leftNode+forwardNode+backwardNode)

    def spin_flip(self, i, j, z):          #This function is meant to just flip the spin of S_ijz passed...
        self.config[i][j][z] = self.config[i][j][z] * -1


    def magnetization(self):            #This function will grab the magnetization of the 3d lattice
        return np.sum(self.config)/ (self.n ** 3)


########################################################################################
# # below is testing out the reconfiguration (aka our 3d class lattice)

# lattice = Lattice(3)       #should make the space of the lattice, and fill it in with spins (+1 / -1)

# print(lattice.config)       #this should show the lattice space

# lattice.spin_flip(1,1,1)                  #flip middle spin of middle lattice

# print("HERE IS THE SPIN FLIP AFTER.....\n\n")
# print(lattice.config, "\n")

# print("HERE IS THE NEIGHBORING_COST for 0,0,0")
# print(lattice.neighboring_cost(0,0,0))    #this would give us the cost, should be through periodic bounds
# print("HERE IS THE NEIGHBORING_COST for 2,2,2")
# print(lattice.neighboring_cost(2,2,2))    #this would give us the cost, should be through periodic bounds
# print("\n")

# print("HERE IS THE magnetization for the lattice")
# print(lattice.magnetization())                   #

