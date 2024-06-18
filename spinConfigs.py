#This file's main purpose is to just configure an Initial Spin Configuration and Spin Flips.

import numpy as np

n = 0   #lattice dimensions nxn

#This function will create our initial lattice space.
def initializeLattice(n):
    setN(n)                      
    lattice = np.zeros((n,n))
    fillLattice(lattice, n)
    return(lattice)

#This function is meant to fill in our initial lattice space with "spins" (up/down)
def fillLattice(lattice, n):
    for x in range(n):
        for y in range(n):
            lattice[x][y] = np.random.choice([1, -1])

#this will be used to store the n size of the lattice
def setN(N):
    global n
    n = N   

#this will be used in other functions where n cant/shouldnt be passed through user input
def getN():
    return n


#This function is meant to calculate the local cost of S_ij 
def neighboringCost(lattice, i, j):
    n=getN()
    currentNode = lattice[i][j]
    aboveNode = lattice[i][(j+1)%n]
    belowNode = lattice[i][(j-1)%n]
    rightNode = lattice[(i+1)%n][j]
    leftNode = lattice[(i-1)%n][j]

    return np.multiply(currentNode, [aboveNode+belowNode+rightNode+leftNode])

#This function is meant to just flip the spin of S_ij passed...
def spinFlip(lattice, i, j):
    lattice[i][j] *= -1


########################################################################################
# below was inital lattice example, to then generalize above...
# lattice = np.zeros((32,32))

# print("Initial Empty Lattice Space")
# print(lattice)

# for x in range(32):
#         for y in range(32):
#             lattice[x][y] = np.random.choice([1, -1])

# print("Initial Lattice Space")
# print(lattice)

########################################################################################
# below was testing of functions. Along with testing of new function neighboringCost
# lattice = initializeLattice(8)

# print(lattice)

# cost = neighboringCost(lattice, 4, 4) #test to see if calculation of cost works
# cost2 = neighboringCost(lattice, 7, 0) #test to see if we have periodic bounds

# print(cost)
# print()
# print(cost2)

########################################################################################
# below is testing just the spin flip...

# lattice = initializeLattice(2)

# print(lattice)

# spinFlip(lattice, 0,0)

# print()

# print(lattice)