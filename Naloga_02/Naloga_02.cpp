#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.1415926f
#endif


// Constants
constexpr const float XjL = -(float)M_PI;
constexpr const float XjU = (float)M_PI;
constexpr const float epsilon = (float)1e-6;

struct Solution {
    std::vector<float> xThetas;
    std::vector<float> xBetas;
    float energy;
    float F;
    float Cr;
};

struct Coordinates {
    float x, y, z;

    Coordinates() : x(0.f), y(0.f), z(0.f) {}
    Coordinates(float xValue, float yValue, float zValue) : x(xValue), y(yValue), z(zValue) {}
};

Coordinates coordinate(std::vector<Coordinates>& allCoords, Solution& element, int i) {
    if (i == 0) return { 0.f, 0.f, 0.f };
    else if (i == 1) return { 0.f, 1.f, 0.f };
    else if (i == 2) return { cosf(element.xThetas[0]), 1.f + sinf(element.xThetas[0]), 0.f };
    else if (3 <= i && i <= element.xThetas.size() + 3) {
        return {
                allCoords[i - 1].x + cosf(element.xThetas[i - 2]) * cosf(element.xBetas[i - 3]),
                allCoords[i - 1].y + sinf(element.xThetas[i - 2]) * cosf(element.xBetas[i - 3]),
                allCoords[i - 1].z + sinf(element.xBetas[i - 3])
        };
    }
    else {
        throw std::runtime_error("Index for calculating coordinate not valid...");
    }
}

float distance(Coordinates& ri, Coordinates& rj) {
    float first = rj.x - ri.x;
    float second = rj.y - ri.y;
    float third = rj.z - ri.z;
    return sqrtf(first * first + second * second + third * third);
}

constexpr float c(char& aminoAcidI, char& aminoAcidJ) {
    if (aminoAcidI == 'A' && aminoAcidJ == 'A') return 1.f;
    if (aminoAcidI == 'B' && aminoAcidJ == 'B') return 0.5f;
    if (aminoAcidI != aminoAcidJ) return (-0.5f);

    throw std::runtime_error("Invalid amino acid pair: " + std::string(1, aminoAcidI) + " and " + std::string(1, aminoAcidJ));
}

float energyCalculation(std::string& aminoAcids, Solution& element) {
    float resultOne = 0.f;
    for (int i = 0; i < element.xThetas.size(); i++) {
        resultOne += (1 - cosf(element.xThetas[i]));
    }
    resultOne /= 4.f;

    std::vector<Coordinates> allCoords(element.xThetas.size() + 2);
    for (int i = 0; i < element.xThetas.size() + 2; i++) {
        allCoords[i] = coordinate(allCoords, element, i);
    }

    float resultTwo = 0.f;
    for (int i = 0; i < element.xThetas.size(); i++) {
        for (int j = i + 2; j < element.xThetas.size() + 2; j++) {
            float inv = distance(allCoords[i], allCoords[j]);
            float inv2 = inv * inv;
            float inv3 = inv * inv2;
            float inv6 = 1 / (inv3 * inv3);
            float inv12 = inv6 * inv6;
            resultTwo += (inv12 - c(aminoAcids[i], aminoAcids[j]) * inv6);
        }
    }
    return resultOne + resultTwo * 4.f;
}

int main(int argc, char* argv[]) {
    std::string aminoAcids;
    unsigned int seed, nFesLmt, runtimeLmt, np;
    unsigned int nFesCounter = 0;
    float target;

    try {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg[0] == 'A' || arg[0] == 'B')) aminoAcids = argv[i];
            else if (arg == "-seed") seed = std::stoi(argv[++i]);
            else if (arg == "-target") target = std::stof(argv[++i]);
            else if (arg == "-nfesLmt") nFesLmt = std::stoi(argv[++i]);
            else if (arg == "-runtimeLmt") runtimeLmt = std::stoi(argv[++i]);
            else if (arg == "-Np" || arg == "-np") np = std::stoi(argv[++i]);
        }

        if (aminoAcids.empty()) throw std::runtime_error("Error: 'aminoAcids' not initialized.");
        if (!seed) throw std::runtime_error("Error: 'seed' not initialized.");
        if (!target) throw std::runtime_error("Error: 'target' not initialized.");
        if (!nFesLmt) throw std::runtime_error("Error: 'nFesLmt' not initialized.");
        if (!runtimeLmt) throw std::runtime_error("Error: 'runtimeLmt' not initialized.");
        if (!np) throw std::runtime_error("Error: 'np' not initialized.");

    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }


    std::mt19937 gen(seed);

    std::uniform_real_distribution<float> float_dist(0.f, 1.f);
    auto randFloat0To1 = [&]() { return float_dist(gen); };

    std::uniform_int_distribution<int> int_dist_np(0, static_cast<int>(np - 1));
    auto randFloatNp = [&]() { return int_dist_np(gen); };

    std::uniform_int_distribution<int> int_dist_d(0, static_cast<int>(aminoAcids.size() - 1));
    auto randIntD = [&]() { return int_dist_d(gen); };

    unsigned int bestEnergy = 0;
    float F, Cr;

    auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

    Solution sol;
    Solution* populationCurrGen = new Solution[np];
    for (unsigned int i = 0; i < np; i++) {
        sol = Solution();
        sol.F = 0.5f;
        sol.Cr = 0.9f;

        sol.xThetas.push_back(XjL + 2 * XjU * randFloat0To1());
        for (int j = 3; j < aminoAcids.size(); j++) {
            sol.xThetas.push_back(XjL + 2 * XjU * randFloat0To1());
            sol.xBetas.push_back(XjL + 2 * XjU * randFloat0To1());
        }

        /* // for testing
        sol.xThetas = {0.7556,  0.0503, -0.8505,  0.0011,  0.2203,  1.1535, -0.1118,  0.1564,    0.1536,  0.0390,  1.2929};
        sol.xBetas = { -0.1156,  0.0230, -1.8169,  2.7985, -3.0959,  -0.3611,  0.4678,  2.2303,  2.9020,  0.1797};
        */

        sol.energy = energyCalculation(aminoAcids, sol);
        populationCurrGen[i] = sol;
        nFesCounter++;
    }

    Solution u, uNew;
    unsigned int r1, r2, r3, rBest, jRand;
    rBest = 0;
    float tmp;
    while (elapsed_ms <= runtimeLmt && nFesCounter <= nFesLmt) {
        now = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

        for (unsigned int i = 0; i < np; i++) {
            u = Solution();

            float randomValue = randFloat0To1();
            if (randomValue < 0.1) F = 0.1f + 0.9f * randomValue;
            else F = populationCurrGen[i].F;

            randomValue = randFloat0To1();
            if (randomValue < 0.1) Cr = randomValue;
            else Cr = populationCurrGen[i].Cr;

            r1 = r2 = r3 = i;
            while (r1 == i) r1 = randFloatNp();
            while (r2 == i || r2 == r1) r2 = randFloatNp();
            while (r3 == i || r3 == r1 || r3 == r2) r3 = randFloatNp();

            jRand = randIntD();
            // then go through for all thetas
            for (int j = 0; j < aminoAcids.size() - 2; j++) {
                if (randFloat0To1() < Cr || j == jRand) {
                    u.xThetas.push_back(populationCurrGen[r3].xThetas[j] + F * (populationCurrGen[r1].xThetas[j] - populationCurrGen[r2].xThetas[j]));
                    if (u.xThetas[j] <= XjL) u.xThetas[j] = 2 * XjU + u.xThetas[j];
                    if (u.xThetas[j] > XjU) u.xThetas[j] = 2 * XjL + u.xThetas[j];
                }
                else u.xThetas.push_back(populationCurrGen[i].xThetas[j]);
            }
            // then go through for all betas
            for (int j = 0; j < aminoAcids.size() - 3; j++) {
                if (randFloat0To1() < Cr || j == jRand) {
                    u.xBetas.push_back(populationCurrGen[r3].xBetas[j] + F * (populationCurrGen[r1].xBetas[j] - populationCurrGen[r2].xBetas[j]));
                    if (u.xBetas[j] <= XjL) u.xBetas[j] = 2 * XjU + u.xBetas[j];
                    if (u.xBetas[j] > XjU) u.xBetas[j] = 2 * XjL + u.xBetas[j];
                }
                else u.xBetas.push_back(populationCurrGen[i].xBetas[j]);
            }
            u.energy = energyCalculation(aminoAcids, u);
            nFesCounter++;

            if (u.energy <= populationCurrGen[i].energy) {
                uNew = Solution();
                for (int j = 0; j < aminoAcids.size() - 2; j++) {
                    uNew.xThetas.push_back(populationCurrGen[r3].xThetas[j] + 0.5f * (u.xThetas[j] - populationCurrGen[i].xThetas[j]));
                    if (uNew.xThetas[j] <= XjL) uNew.xThetas[j] = 2 * XjU + uNew.xThetas[j];
                    if (uNew.xThetas[j] > XjU) uNew.xThetas[j] = 2 * XjL + uNew.xThetas[j];
                }
                for (int j = 0; j < aminoAcids.size() - 3; j++) {
                    uNew.xBetas.push_back(populationCurrGen[r3].xBetas[j] + 0.5f * (u.xBetas[j] - populationCurrGen[i].xBetas[j]));
                    if (uNew.xBetas[j] <= XjL) uNew.xBetas[j] = 2 * XjU + uNew.xBetas[j];
                    if (uNew.xBetas[j] > XjU) uNew.xBetas[j] = 2 * XjL + uNew.xBetas[j];
                }

                uNew.energy = energyCalculation(aminoAcids, uNew);
                nFesCounter++;

                if (uNew.energy <= u.energy) {
                    populationCurrGen[i].xThetas = uNew.xThetas;
                    populationCurrGen[i].xBetas = uNew.xBetas;
                    populationCurrGen[i].energy = uNew.energy;
                    populationCurrGen[i].Cr = Cr;
                    populationCurrGen[i].F = F;
                }
                else {
                    populationCurrGen[i].xThetas = u.xThetas;
                    populationCurrGen[i].xBetas = u.xBetas;
                    populationCurrGen[i].energy = u.energy;
                }
            }
        }

        tmp = populationCurrGen[0].energy;
        for (unsigned int j = 1; j < np; j++) {
            if (tmp > populationCurrGen[j].energy) {
                rBest = j;
                tmp = populationCurrGen[j].energy;
            }
        }
        bestEnergy = rBest;

        if (populationCurrGen[rBest].energy <= target + epsilon && populationCurrGen[rBest].energy >= target + epsilon) break;
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << populationCurrGen[bestEnergy].energy << '\n';
    std::cout << nFesCounter << '\n';
    std::cout << duration << '\n';
    //std::cout << "Speed (nFes / duration = speed): " << nFesCounter << " / " << duration << " = " << nFesCounter/duration << '\n';
    delete[] populationCurrGen;



    return 0;
}