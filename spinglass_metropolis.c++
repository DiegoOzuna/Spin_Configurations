#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib> // for atoi

//for data saving...
#include <json.hpp> //This header file will need to be installed from here.... <https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp>
#include <fstream>

//for parallel code...
#include <thread>
#include <mutex>
#include <memory> //for std::shared_ptr

std::mutex mtx;

// Initialize the random number generator
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0,1.0);

constexpr int L = 9; // The size of the lattice/bonds

class Lattice {
public:
    int spins[L][L][L];

    Lattice() {
        for(int i = 0; i < L; i++) {
            for(int j = 0; j < L; j++) {
                for(int k = 0; k < L; k++) {
                    spins[i][j][k] = (distribution(generator) < 0.5) ? 1 : -1;
                }
            }
        }
    }
};

class Bond {
public:
    int config[L][L][L];

    Bond() {
        for(int i = 0; i < L; i++) {
            for(int j = 0; j < L; j++) {
                for(int k = 0; k < L; k++) {
                    config[i][j][k] = (distribution(generator) < 0.5) ? 1 : -1;
                }
            }
        }
    }
};


void single_spin_flips(Lattice &lattice, Bond &bond, double temp) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    // Generate a list of random numbers of size n*n*n
    double random_numbers[L*L*L];
    for (int i = 0; i < L*L*L; ++i) {
        random_numbers[i] = distribution(generator);
    }

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int z = 0; z < L; ++z) {
                int idx = i * L * L + j * L + z;
                int up = lattice.spins[(i-1+L)%L][j][z] * bond.config[(i-1+L)%L][j][z];
                int down = lattice.spins[(i+1)%L][j][z] * bond.config[(i+1)%L][j][z];
                int left = lattice.spins[i][(j-1+L)%L][z] * bond.config[i][(j-1+L)%L][z];
                int right = lattice.spins[i][(j+1)%L][z] * bond.config[i][(j+1)%L][z];
                int front = lattice.spins[i][j][(z-1+L)%L] * bond.config[i][j][(z-1+L)%L];
                int back = lattice.spins[i][j][(z+1)%L] * bond.config[i][j][(z+1)%L];
                double energy_of_flip = -2 * lattice.spins[i][j][z] * (up + down + left + right + front + back);

                if (random_numbers[idx] < std::exp(-energy_of_flip / temp)) {
                    lattice.spins[i][j][z] *= -1;
                }
            }
        }
    }
}


double TotalEnergy(Lattice &lattice, Bond &bonds) {
    double total_energy = 0.0;

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                // Calculate the energy contribution from each spin and its neighbors
                double E = -lattice.spins[i][j][k] * (
                    lattice.spins[(i+1)%L][j][k] * bonds.config[(i+1)%L][j][k] +
                    lattice.spins[i][(j+1)%L][k] * bonds.config[i][(j+1)%L][k] +
                    lattice.spins[i][j][(k+1)%L] * bonds.config[i][j][(k+1)%L]
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

    Bond bonds;

    std::vector<Lattice> latticeConfigurations1(temp_steps.size(), Lattice());
    std::vector<Lattice> latticeConfigurations2(temp_steps.size(), Lattice());

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

        if (z % step == 0 && z >= equilock) {
            for (size_t t = 0; t < temp_steps.size(); ++t) {
                lattice1_dict[temp_steps[t]].push_back(latticeConfigurations1[t]);
                lattice2_dict[temp_steps[t]].push_back(latticeConfigurations2[t]);
            }
        }
    }

    return std::make_pair(lattice1_dict, lattice2_dict);
}


void perform_operations(int config, int L, int MonteCarloSteps, std::vector<double>& temp_steps, int measureEvery, std::shared_ptr<nlohmann::json> j) {
    auto [lattice1_dict, lattice2_dict] = montecarlo(L, MonteCarloSteps, temp_steps, measureEvery, 10000);

    //LOCK the mutex before update of json...
    std::lock_guard<std::mutex> lock(mtx);

    // Add data of S1 to the JSON object...
    for (const auto& pair : lattice1_dict) {
        double temp = pair.first;
        const std::vector<Lattice>& lattices = pair.second;

        for (const Lattice& lattice : lattices) {
            (*j)["Configuration"][std::to_string(config)]["S1"]["Temp"][std::to_string(temp)].push_back(lattice.spins);
        }
    }
    // Add data of S2 to the JSON object...
    for (const auto& pair : lattice2_dict) {
        double temp = pair.first;
        const std::vector<Lattice>& lattices = pair.second;

        for (const Lattice& lattice : lattices) {
            (*j)["Configuration"][std::to_string(config)]["S2"]["Temp"][std::to_string(temp)].push_back(lattice.spins);
        }
    }
    // mtx.unlock();
}

int main(int argc, char* argv[]) {
    if(argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <maxTemp> <MonteCarloSteps> <measureEvery> <num_disorder_configs> \n";
        return 1;
    }

    double maxTemp = std::atoi(argv[1]);
    int MonteCarloSteps = std::atoi(argv[2]);
    int measureEvery = std::atoi(argv[3]);
    int num_disorder_configs = std::atoi(argv[4]);

    std::vector<double> temp_steps;
    for (double temp = 0.8; temp <= maxTemp; temp += 0.1) {
        temp_steps.push_back(temp);
    }

    // Change the type of 'j' to std::shared_ptr<nlohmann::json>
    std::shared_ptr<nlohmann::json> j = std::make_shared<nlohmann::json>();

    // Create a vector to hold the threads
    std::vector<std::thread> threads;

    for (int config = 0; config < num_disorder_configs; ++config) {
        // Start a new thread for each operation
        threads.push_back(std::thread(perform_operations, std::ref(config), std::ref(L), std::ref(MonteCarloSteps), std::ref(temp_steps), std::ref(measureEvery), std::ref(j)));

    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // Write JSON to file
    std::ofstream file("data.json");
    file << (*j).dump(4);

    return 0;
}

