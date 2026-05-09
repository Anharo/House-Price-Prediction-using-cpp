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
double R2Score(const vector<double>& y_true,const vector<double>& y_pred) {
    double mean_y=0.0;
    int m= y_true.size();
    for (double val : y_true) {
        mean_y += val;
    }
    mean_y /= m;
    double ss_tot=0.0, ss_res=0.0;
    for(int i=0; i<m; i++) {
        ss_tot += (y_true[i] -mean_y) * (y_true[i] -mean_y);
        ss_res += (y_true[i] -y_pred[i]) * (y_true[i] -y_pred[i]);
    }
    return 1.0 - (ss_res / ss_tot);
}

double MAE(const vector<double>& y_true,const vector<double>& y_pred) {
    double sum = 0.0;
    int m = y_true.size();
    for (int i = 0; i < m; i++) {
        sum += abs(y_true[i] - y_pred[i]);
    }
    return sum / m;
}