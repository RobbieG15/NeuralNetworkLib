#include "activation.hpp"
#include <cmath>

class Tanh : public Activation {
public:
    double activate(double x) const override {
        return std::tanh(x);
    }

    double derivative(double x) const override {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
};
