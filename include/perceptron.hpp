#pragma once
#include "vector.hpp"
#include "activation_funcs/activation.hpp"

class Perceptron {
private:
    Vector weights;
    double bias;
    double lr;
    Activation* activation;

public:
    Perceptron(size_t input_dim, Activation* act, double learning_rate = 0.01);

    double forward(const Vector& x) const;
    void train(const Vector& x, double target);
};
