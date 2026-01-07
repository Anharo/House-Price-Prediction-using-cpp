#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

bool Dataset::loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return false;
    }

    std::string line;
    getline(file, line); // skip header

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }

    file.close();
    return true;
}

// Normalize target (price)
void Dataset::normalize() {
    int m = X.size();
    int n = X[0].size();

    mean.resize(n, 0.0);
    stddev.resize(n, 0.0);

    // Feature mean
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++)
            mean[j] += X[i][j];
        mean[j] /= m;
    }

    // Feature std
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++)
            stddev[j] += pow(X[i][j] - mean[j], 2);
        stddev[j] = sqrt(stddev[j] / m);
    }

    // Normalize features
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            X[i][j] = (X[i][j] - mean[j]) / stddev[j];

    // 🔥 Normalize target y
    for (double val : y)
        y_mean += val;
    y_mean /= m;

    for (double val : y)
        y_std += pow(val - y_mean, 2);
    y_std = sqrt(y_std / m);

    for (double& val : y)
        val = (val - y_mean) / y_std;
}
double Dataset::denormalizeY(double value) const {
    return value * y_std + y_mean;
}
std::vector<double> Dataset::normalizeInput(
    const std::vector<double>& input) const {

    std::vector<double> normalized = input;

    for (int j = 0; j < input.size(); j++) {
        normalized[j] = (input[j] - mean[j]) / stddev[j];
    }

    return normalized;
}
std::pair<Dataset, Dataset> Dataset::trainTestSplit(double test_ratio) const {
    Dataset train, test;

    int m = X.size();
    int test_size = static_cast<int>(m * test_ratio);

    std::vector<int> indices(m);
    for (int i = 0; i < m; i++)
        indices[i] = i;

    // Shuffle indices (fixed seed = reproducible split)
    std::shuffle(indices.begin(), indices.end(), std::mt19937(42));

    for (int i = 0; i < m; i++) {
        if (i < test_size) {
            test.X.push_back(X[indices[i]]);
            test.y.push_back(y[indices[i]]);
        } else {
            train.X.push_back(X[indices[i]]);
            train.y.push_back(y[indices[i]]);
        }
    }

    // 🔥 IMPORTANT: copy normalization parameters
    train.mean   = mean;
    train.stddev = stddev;
    train.y_mean = y_mean;
    train.y_std  = y_std;

    test.mean   = mean;
    test.stddev = stddev;
    test.y_mean = y_mean;
    test.y_std  = y_std;

    return {train, test};
}
