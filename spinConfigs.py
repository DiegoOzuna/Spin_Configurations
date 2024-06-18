#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.
#It is also meant to handle operations done on the lattice itself.

import numpy as np

class Lattice:
    def __init__(self, n):      #This function will create our initial lattice space.
        self.n = n
        self.config = np.zeros((n,n))
        self.fill_lattice()

    def fill_lattice(self):     #This function is meant to fill in our initial lattice space with "spins" (up/down)
        for x in range(self.n):
            for y in range(self.n):
                self.config[x][y] = np.random.choice([1, -1])

    def neighboring_cost(self, i, j):       #This function is meant to calculate the local cost of S_ij 
        currentNode = self.config[i][j]
        aboveNode = self.config[i][(j+1)%self.n]
        belowNode = self.config[i][(j-1)%self.n]
        rightNode = self.config[(i+1)%self.n][j]
        leftNode = self.config[(i-1)%self.n][j]

        return np.multiply(currentNode, [aboveNode+belowNode+rightNode+leftNode])

    def spin_flip(self, i, j):          #This function is meant to just flip the spin of S_ij passed...
        self.config[i][j] *= -1


########################################################################################
# # below is testing out the reconfiguration (aka our class lattice)

# lattice = Lattice(3)       #should make the space of the lattice, and fill it in with spins (+1 / -1)

# print(lattice.config)       #this should show the lattice space

# print(lattice.neighboring_cost(0,0))    #this would give us the cost, should be through periodic bounds
# print(lattice.neighboring_cost(2,2))    #this would give us the cost, should be through periodic bounds

# lattice.spin_flip(1,1)                  #flip middle spin

# print(lattice.config)                   #should see middle spin be opposite

