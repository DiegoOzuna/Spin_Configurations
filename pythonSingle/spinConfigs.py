#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.
#It is also meant to handle operations done on the lattice itself.

import numpy as np
import copy

class Lattice:
    def __init__(self, n, config=None):  #This function will create our initial 3D lattice space. given bonds and lattice is optional
        self.n = n
        if config is None:
            self.config = self.fill_lattice()
        else:
              self.config = config #Use provided lattice configuration


    def fill_lattice(self):     #This function is meant to fill in our initial lattice space with "spins" (up/down)
        config = np.random.choice([1, -1], size=(self.n, self.n, self.n))
        return config

    def neighboring_cost(self, bonds, i, j, z):    #This function is meant to calculate the local cost of S_ijz
        currentNode = self.config[i][j][z]
        #The local cost is the summation of all the neighboring spins * their weight with respect to the current node
        #multiplied by the current node...
        # Calculate indices for neighbors

        # This is within the same lattice space... (2d)
        above = (i, (j+1)%self.n, z)
        below = (i, (j-1)%self.n, z)
        right = ((i+1)%self.n, j, z)
        left = ((i-1)%self.n, j, z)

        # This is entering the other lattice space... (3d)
        forward = (i, j, (z+1)%self.n)
        backward = (i, j, (z-1)%self.n)

        # Calculate bond values for neighbors
        neighbors = [above, below, right, left, forward, backward]
        bond_values = [self.config[n]* bonds.config[n] for n in neighbors]

        return currentNode * sum(bond_values)

    def spin_flip(self, i, j, z):          #This function is meant to just flip the spin of S_ijz passed...
        self.config[i][j][z] = self.config[i][j][z] * -1


    def magnetization(self, bonds):            #This function will grab the magnetization of the 3d lattice
        return np.sum(np.multiply(self.config, bonds.config))/ (self.n ** 3)
    

class Bonds:
    def __init__(self, n, config=None):  #This function will create our initial bond.
        self.n = n
        if config is None:
            self.config = self.fill_bonds()  # Generate bonds if not provided
        else:
            self.config = config             #if we determine configuration outside of bonds, set value.
    
    def fill_bonds(self):       #This function is meant to fill in the values for the bonds between 
        bonds = np.random.choice([1,-1], size=(self.n, self.n, self.n))
        return bonds


########################################################################################
#below is testing out the reconfiguration (aka our 3d class lattice)

# lattice = Lattice(3)       #should make the space of the lattice, and fill it in with spins (+1 / -1)

# print("Initial lattice configurations \n")

# print(lattice.config)       #this should show the lattice space

# print("BONDS FOR THIS LATTICE \n")

# print(lattice.bonds)        #this should show the bonds made for the lattice


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
# print("\n")


# print("TESTING FOR SAME BONDS BUT DIFFERENT SPINS!\n")
# replica = Lattice(8, lattice.bonds)
# count = 0


# def check_bonds(lattice, replica):
#     for x in range(lattice.n):
#         for y in range(lattice.n):
#             for z in range(lattice.n):
#                 if lattice.bonds[x][y][z] != replica.bonds[x][y][z]:
#                     return False

#     return True

# def check_spins(lattice, replica):
#     for x in range(lattice.n):
#         for y in range(lattice.n):
#             for z in range(lattice.n):
#                 if lattice.config[x][y][z] != replica.config[x][y][z]:
#                     return False

#     return True

# print("Checking bonds are equal....", check_bonds(lattice, replica))
# print("Checking Spin Configurations....", check_spins(lattice, replica))

