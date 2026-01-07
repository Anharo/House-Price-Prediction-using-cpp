#include "metrics.h"
#include <cmath>

double RMSE(const std::vector<double>& y_true,
            const std::vector<double>& y_pred) {

    double sum = 0.0;
    int m = y_true.size();

    for (int i = 0; i < m; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }

    return std::sqrt(sum / m);
}
