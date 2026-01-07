#include <iostream>
#include <vector>
#include "dataset.h"
#include "linear_regression.h"
#include <algorithm>  

int main() {
    Dataset data;
    if (!data.loadCSV("data/house_prices.csv")) {
        return 1;
    }
    data.normalize();

    int n_features = data.X[0].size();

    // ===== TRAIN MODEL =====
    LinearRegression model(n_features);
    model.gradientDescent(data.X, data.y, 0.01, 5000);

    std::cout << "Training completed.\n";

    // ===== SAVE MODEL =====
    model.saveModel("model.txt");
    std::cout << "Model saved to model.txt\n";

    // ===== LOAD MODEL =====
    LinearRegression loadedModel(n_features);
    loadedModel.loadModel("model.txt");
    std::cout << "Model loaded successfully.\n";

    // ===== USER INPUT LOOP =====
while (true) {
    std::vector<double> house(n_features);

    std::cout << "\nEnter house details:\n";

    std::cout << "Area (sqft, 200-10000): ";
    std::cin >> house[0];
    house[0] = std::clamp(house[0], 200.0, 10000.0);

    std::cout << "Bedrooms (1-10): ";
    std::cin >> house[1];
    house[1] = std::clamp(house[1], 1.0, 10.0);

    std::cout << "Bathrooms (1-10): ";
    std::cin >> house[2];
    house[2] = std::clamp(house[2], 1.0, 10.0);

    std::cout << "Location score (1-10): ";
    std::cin >> house[3];
    house[3] = std::clamp(house[3], 1.0, 10.0);

    std::cout << "Age of house (0-100 years): ";
    std::cin >> house[4];
    house[4] = std::clamp(house[4], 0.0, 100.0);

    // Normalize input
    std::vector<double> normalizedHouse = data.normalizeInput(house);

    // Predict
    double pred_norm = loadedModel.predict(normalizedHouse);
    double price = data.denormalizeY(pred_norm);

    std::cout << "\nPredicted House Price (INR): " << price << "\n";

    char choice;
    std::cout << "\nPredict another house? (y/n): ";
    std::cin >> choice;
    if (choice != 'y' && choice != 'Y') {
        break;
    }
}


    std::cout << "\nExiting program.\n";
    return 0;
}
