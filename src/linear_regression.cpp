#include "linear_regression.h"
#include <cmath>
#include <iostream>
#include <fstream>
// Constructor
LinearRegression::LinearRegression(int n_features) {
    theta.resize(n_features + 1, 0.0); // +1 for bias
}

// Prediction
double LinearRegression::predict(const std::vector<double>& x) {
    double result = theta[0]; // bias
    for (int i = 0; i < x.size(); i++) {
        result += theta[i + 1] * x[i];
    }
    return result;
}

// Cost function (MSE)
double LinearRegression::computeCost(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y) {

    int m = X.size();
    double cost = 0.0;

    for (int i = 0; i < m; i++) {
        double error = predict(X[i]) - y[i];
        cost += error * error;
    }

    return cost / (2 * m);
}

// Gradient Descent
void LinearRegression::gradientDescent(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double alpha,
    int epochs) {

    int m = X.size();
    int n = X[0].size();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::vector<double> gradients(n + 1, 0.0);

        for (int i = 0; i < m; i++) {
            double error = predict(X[i]) - y[i];

            gradients[0] += error;
            for (int j = 0; j < n; j++) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        for (int j = 0; j <= n; j++) {
            theta[j] -= (alpha / m) * gradients[j];
        }

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Cost: " << computeCost(X, y)
                      << "\n";
        }
    }
}
void LinearRegression::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    for (double t : theta)
        file << t << " ";
    file.close();
}

void LinearRegression::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    for (double& t : theta)
        file >> t;
    file.close();
}