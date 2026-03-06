#include <iostream>
#include <vector>
#include <algorithm>
#include "dataset.h"
#include "linear_regression.h"

using namespace std;

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

    cout << "Training completed.\n";

    // ===== SAVE MODEL =====
    model.saveModel("model.txt");
    cout << "Model saved to model.txt\n";

    // ===== LOAD MODEL =====
    LinearRegression loadedModel(n_features);
    loadedModel.loadModel("model.txt");
    cout << "Model loaded successfully.\n";

    // ===== USER INPUT LOOP =====
    while (true) {

        vector<double> house(n_features);

        cout << "\nEnter house details:\n";

        cout << "Area (sqft, 200-10000): ";
        cin >> house[0];
        house[0] = clamp(house[0], 200.0, 10000.0);

        cout << "Bedrooms (1-10): ";
        cin >> house[1];
        house[1] = clamp(house[1], 1.0, 10.0);

        cout << "Bathrooms (1-10): ";
        cin >> house[2];
        house[2] = clamp(house[2], 1.0, 10.0);

        cout << "Location score (1-10): ";
        cin >> house[3];
        house[3] = clamp(house[3], 1.0, 10.0);

        cout << "Age of house (0-100 years): ";
        cin >> house[4];
        house[4] = clamp(house[4], 0.0, 100.0);

        // Normalize input
        vector<double> normalizedHouse = data.normalizeInput(house);

        // Predict
        double pred_norm = loadedModel.predict(normalizedHouse);
        double price = data.denormalizeY(pred_norm);

        cout << "\nPredicted House Price (INR): " << price << "\n";

        char choice;
        cout << "\nPredict another house? (y/n): ";
        cin >> choice;

        if (choice != 'y' && choice != 'Y') {
            break;
        }
    }

    cout << "\nExiting program.\n";

    return 0;
}