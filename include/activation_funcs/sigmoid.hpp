#include "activation.hpp"
#include <cmath>

class Sigmoid : public Activation {
public:
    double activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double x) const override {
        double s = activate(x);
        return s * (1.0 - s);
    }
};
