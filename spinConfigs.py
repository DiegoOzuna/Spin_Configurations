#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.
#It is also meant to handle operations done on the lattice itself.

import numpy as np
import copy

class Lattice:
    def __init__(self, n):      #This function will create our initial 3D lattice space.
        self.n = n
        self.config = self.fill_lattice()
        self.bonds = self.fill_bonds()

    def genReplica(self):         #This function will replicate our current lattice's bonds but with a different spin configuration
        replica = copy.deepcopy(self)  # Create a deep copy of the current instance
        replica.config = self.fill_lattice()  # Generate a new lattice configuration
        return replica

    def fill_lattice(self):     #This function is meant to fill in our initial lattice space with "spins" (up/down)
        config = np.random.choice([1, -1], size=(self.n, self.n, self.n))
        return config
    
    def fill_bonds(self):       #This function is meant to fill in the values for the bonds between 
        bonds = np.random.choice([1,-1], size=(self.n, self.n, self.n))
        return bonds


    def neighboring_cost(self, i, j, z):       #This function is meant to calculate the local cost of S_ijz
        currentNode = self.config[i][j][z]
        #The local cost is the summation of all the neighboring spins * their weight with respect to the current node
        #multiplied by the current node...

        # This is within the same lattice space... (2d)
        aboveNode = self.config[i][(j+1)%self.n][z] * self.bonds[i][(j+1)%self.n][z] #This is the bond between currentNode and aboveNode...
        belowNode = self.config[i][(j-1)%self.n][z] * self.bonds[i][(j-1)%self.n][z] #This is the bond between currentNode and belowNode...
        rightNode = self.config[(i+1)%self.n][j][z] *  self.bonds[(i+1)%self.n][j][z] #This is the bond between currentNode and rightNode...
        leftNode = self.config[(i-1)%self.n][j][z] * self.bonds[(i-1)%self.n][j][z] #This is the bond between currentNode and leftNode...

        # This is entering the other lattice space... (3d)
        forwardNode = self.config[i][j][(z+1)%self.n] * self.bonds[i][j][(z+1)%self.n] #This is the bond between currentNode and fowardNode...
        backwardNode = self.config[i][j][(z-1)%self.n] * self.bonds[i][j][(z-1)%self.n] #This is the bond between currentNode and backwardNode

        return currentNode * (aboveNode+belowNode+rightNode+leftNode+forwardNode+backwardNode)

    def spin_flip(self, i, j, z):          #This function is meant to just flip the spin of S_ijz passed...
        self.config[i][j][z] = self.config[i][j][z] * -1


    def magnetization(self):            #This function will grab the magnetization of the 3d lattice
        return np.sum(np.multiply(self.config, self.bonds))/ (self.n ** 3)


########################################################################################
# below is testing out the reconfiguration (aka our 3d class lattice)

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
# replica = lattice.genReplica()
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

