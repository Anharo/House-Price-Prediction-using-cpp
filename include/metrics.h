#ifndef METRICS_H
#define METRICS_H

#include <vector>

double RMSE(const std::vector<double>& y_true,
            const std::vector<double>& y_pred);

#endif
