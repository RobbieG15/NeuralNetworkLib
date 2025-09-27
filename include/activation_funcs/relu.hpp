#include "activation.hpp"

class ReLU : public Activation {
public:
    double activate(double x) const override {
        return x > 0.0 ? x : 0.0;
    }

    double derivative(double x) const override {
        return x > 0.0 ? 1.0 : 0.0;
    }
};
