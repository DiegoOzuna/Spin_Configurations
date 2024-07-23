#include <iostream>
#include <random>
#include <vector>
#include <map>
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

void single_spin_flips(Lattice &lattice, Bond &bond, double temp) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int n = lattice.L;
    double oneDimSize = pow(n, 3);
    // Generate a list of random numbers of size n*n*n
    std::vector<double> random_numbers(oneDimSize);
    for (int i = 0; i < oneDimSize; ++i) {
        random_numbers[i] = distribution(generator);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int z = 0; z < n; ++z) {
                int idx = i * n * n + j * n + z;
                double energy_of_flip = 2 * energy(lattice, bond, i, j, z);
                if (random_numbers[idx] < std::exp(-energy_of_flip / temp)) {
                    lattice.spins[i][j][z] *= -1;
                }
            }
        }
    }
}

double TotalEnergy(Lattice &lattice, Bond &bonds) {
    double total_energy = 0.0;
    int nx = lattice.L;
    int ny = lattice.L;
    int nz = lattice.L;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                // Calculate the energy contribution from each spin and its neighbors
                double E = -lattice.spins[i][j][k] * (
                    lattice.spins[(i+1)%nx][j][k] * bonds.config[(i+1)%nx][j][k] +
                    lattice.spins[i][(j+1)%ny][k] * bonds.config[i][(j+1)%ny][k] +
                    lattice.spins[i][j][(k+1)%nz] * bonds.config[i][j][(k+1)%nz]
                );
                // Add the energy contribution to the total energy
                total_energy += E;
            }
        }
    }

    return total_energy;
}


std::pair<std::map<double, std::vector<Lattice> >, std::map<double, std::vector<Lattice> > > montecarlo(int L, int MCS, std::vector<double> temp_steps, int step, int equilock) {
    std::map<double, std::vector<Lattice> > lattice1_dict;
    std::map<double, std::vector<Lattice> > lattice2_dict;

    Bond bonds(L);

    std::vector<Lattice> latticeConfigurations1(temp_steps.size(), Lattice(L));
    std::vector<Lattice> latticeConfigurations2(temp_steps.size(), Lattice(L));

    std::vector<double> Elist1(temp_steps.size(), 0.0);
    std::vector<double> Elist2(temp_steps.size(), 0.0);

    for (int z = 0; z < MCS; ++z) {
        for (size_t t = 0; t < temp_steps.size(); ++t) {
            single_spin_flips(latticeConfigurations1[t], bonds, temp_steps[t]);
            single_spin_flips(latticeConfigurations2[t], bonds, temp_steps[t]);
        }

        for (size_t t = 0; t < temp_steps.size(); ++t) {
            Elist1[t] = TotalEnergy(latticeConfigurations1[t], bonds);
            Elist2[t] = TotalEnergy(latticeConfigurations2[t], bonds);
        }

        for (size_t t = 0; t < temp_steps.size() - 1; ++t) {
            double temp1 = std::exp((Elist1[t] - Elist1[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]));
            double temp2 = std::exp((Elist2[t] - Elist2[t+1]) * (1/temp_steps[t] - 1/temp_steps[t+1]));

            std::default_random_engine generator;
            std::uniform_real_distribution<double> distribution(0.0,1.0);

            if (distribution(generator) < temp1) {
                std::swap(latticeConfigurations1[t], latticeConfigurations1[t+1]);
                std::swap(Elist1[t], Elist1[t+1]);
            }

            if (distribution(generator) < temp2) {
                std::swap(latticeConfigurations2[t], latticeConfigurations2[t+1]);
                std::swap(Elist2[t], Elist2[t+1]);
            }
        }

        if (z % step == 0 && z > equilock) {
            for (size_t t = 0; t < temp_steps.size(); ++t) {
                lattice1_dict[temp_steps[t]].push_back(latticeConfigurations1[t]);
                lattice2_dict[temp_steps[t]].push_back(latticeConfigurations2[t]);
            }
        }
    }

    return std::make_pair(lattice1_dict, lattice2_dict);
}

int main(int argc, char* argv[]) {
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <lattice size> <maxTemp> <MonteCarloSteps> <measureEvery> <num_disorder_configs> \n";
        return 1;
    }

    int L = std::atoi(argv[1]);
    double maxTemp = std::atof(argv[2]);
    int MonteCarloSteps = std::atoi(argv[3]);
    int measureEvery = std::atoi(argv[4]);
    int num_disorder_configs = std::atoi(argv[5]);

    std::vector<double> temp_steps;
    for (double temp = 0.8; temp < maxTemp; temp += 0.1) {
        temp_steps.push_back(temp);
    }

    auto [lattice1_dict, lattice2_dict] = montecarlo(L, MonteCarloSteps, temp_steps, measureEvery, MonteCarloSteps/2);


    return 0;
}
