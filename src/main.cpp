#include <iostream>
#include <vector>
#include <algorithm>
#include "dataset.h"
#include "linear_regression.h"
#include "metrics.h"         

using namespace std;

int main() {

    Dataset data;

    if (!data.loadCSV("../data/house_prices.csv")) {
        return 1;
    }

    data.normalize();

    // ===== TRAIN / TEST SPLIT ===== 
    auto [train, test] = data.trainTestSplit(0.2);  

    int n_features = train.X[0].size();

    // ===== TRAIN MODEL on train set =====
    LinearRegression model(n_features);
    model.gradientDescent(train.X, train.y, 0.01, 5000);
    cout << "\nTraining completed.\n";

    // ===== EVALUATE on test set =====
    vector<double> y_pred;
    for (const auto& x : test.X) {
        y_pred.push_back(model.predict(x));
    }

    // Denormalize for real-world metrics
    vector<double> y_pred_real, y_true_real;
    for (int i = 0; i < test.y.size(); i++) {
        y_pred_real.push_back(data.denormalizeY(y_pred[i]));
        y_true_real.push_back(data.denormalizeY(test.y[i]));
    }

    cout << "\n===== Model Evaluation (Test Set) =====\n";
    cout << "RMSE   : " << RMSE(y_true_real, y_pred_real)    << " INR\n";
    cout << "MAE    : " << MAE(y_true_real, y_pred_real)     << " INR\n";
    cout << "R²     : " << R2Score(y_true_real, y_pred_real) << "\n";
    cout << "========================================\n";

    // ===== SAVE / LOAD MODEL =====
    model.saveModel("model.txt");
    cout << "\nModel saved to model.txt\n";

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

        vector<double> normalizedHouse = data.normalizeInput(house);

        double pred_norm = loadedModel.predict(normalizedHouse);
        double price = data.denormalizeY(pred_norm);

        cout << "\nPredicted House Price (INR): " << price << "\n";

        char choice;
        cout << "\nPredict another house? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y') break;
    }

    cout << "\nExiting program.\n";
    return 0;
}