#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <utility>

class Dataset {
public:
    // Data
    std::vector<std::vector<double>> X; // features
    std::vector<double> y;              // target

    // Core methods
    bool loadCSV(const std::string& filename);
    void normalize();

    // Utility methods
    std::vector<double> normalizeInput(const std::vector<double>& input) const;
    double denormalizeY(double value) const;

    // 🔥 Train / Test split
    std::pair<Dataset, Dataset> trainTestSplit(double test_ratio) const;

private:
    // Feature normalization parameters
    std::vector<double> mean;
    std::vector<double> stddev;

    // Target normalization parameters
    double y_mean = 0.0;
    double y_std  = 1.0;
};

#endif
