#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>
class LinearRegression {
public:
    LinearRegression(int n_features);

    double predict(const std::vector<double>& x);
    double computeCost(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y);

    void gradientDescent(const std::vector<std::vector<double>>& X,
                         const std::vector<double>& y,
                         double alpha,
                         int epochs);
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    std::vector<double> theta;
};

#endif
