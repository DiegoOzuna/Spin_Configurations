#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstdlib> // for atoi

// Initialize the random number generator
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0,1.0);

// Lattice class that self initializes itself with 1 or -1
class Lattice {
    public:
        int L;
        std::vector<std::vector<std::vector<int> > > spins;

        Lattice(int size) : L(size) {
            spins.resize(L, std::vector<std::vector<int> >(L, std::vector<int>(L)));
            for(int i = 0; i < L; i++) {
                for(int j = 0; j < L; j++) {
                    for(int k = 0; k < L; k++) {
                        spins[i][j][k] = (distribution(generator) < 0.5) ? 1 : -1;
                    }
                }
            }
        }
    };

// Bond class that self initializes itself with 1 or -1
class Bond {
    public:
        int L;
        std::vector<std::vector<std::vector<int> > > config;

        Bond(int size) : L(size) {
            config.resize(L, std::vector<std::vector<int> >(L, std::vector<int>(L)));
            for(int i = 0; i < L; i++) {
                for(int j = 0; j < L; j++) {
                    for(int k = 0; k < L; k++) {
                        config[i][j][k] = (distribution(generator) < 0.5) ? 1 : -1;
                    }
                }
            }
        }
    };

// Function to calculate the energy of a spin
int energy(Lattice &lattice, Bond &bond, int i, int j, int k) {
    int L = lattice.L;
    int up = lattice.spins[(i-1+L)%L][j][k] * bond.config[(i-1+L)%L][j][k];
    int down = lattice.spins[(i+1)%L][j][k] * bond.config[(i+1)%L][j][k];
    int left = lattice.spins[i][(j-1+L)%L][k] * bond.config[i][(j-1+L)%L][k];
    int right = lattice.spins[i][(j+1)%L][k] * bond.config[i][(j+1)%L][k];
    int front = lattice.spins[i][j][(k-1+L)%L] * bond.config[i][j][(k-1+L)%L];
    int back = lattice.spins[i][j][(k+1)%L] * bond.config[i][j][(k+1)%L];
    
    return -lattice.spins[i][j][k] * (up + down + left + right + front + back);
}


int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <lattice size>\n";
        return 1;
    }
    int L = std::atoi(argv[1]);
    Lattice lattice1(L);
    Lattice lattice2(L);
    Bond bond(L);

    std::cout << "lattice 1\n";
    // Print the lattice
    for(int i = 0; i < L; i++) {
        for(int j = 0; j < L; j++) {
            for(int k = 0; k < L; k++) {
                std::cout << lattice1.spins[i][j][k] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    std::cout << "lattice 2\n";
    // Print the lattice
    for(int i = 0; i < L; i++) {
        for(int j = 0; j < L; j++) {
            for(int k = 0; k < L; k++) {
                std::cout << lattice2.spins[i][j][k] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    // Print the bonds
    std::cout << "bond \n";
    for(int i = 0; i < L; i++) {
        for(int j = 0; j < L; j++) {
            for(int k = 0; k < L; k++) {
                std::cout << bond.config[i][j][k] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    std::cout << "Energy at (1,1,1) for lattice 1: " << energy(lattice1, bond, 1,1,1);
    std::cout << '\n';
    std::cout << "Energy at (1,1,1) for lattice 2: " << energy(lattice2, bond, 1,1,1);
    std::cout << '\n';


    // Commented out for now
    // for(int i = 0; i < 10000; i++) {
    //     metropolis(lattice);
    // }

    return 0;
}
