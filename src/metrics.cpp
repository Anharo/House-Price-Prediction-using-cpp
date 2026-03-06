#include "metrics.h"
#include <cmath>
#include <vector>
using namespace std;

double RMSE(const vector<double>& y_true,const vector<double>& y_pred) {
    double sum = 0.0;
    int m = y_true.size();

    for (int i = 0; i < m; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sqrt(sum / m);
}