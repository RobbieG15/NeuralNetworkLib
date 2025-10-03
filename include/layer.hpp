#pragma once
#include "matrix.hpp"
#include "vector.hpp"
#include "activation_funcs/activation.hpp"

class Layer {
private:
    Matrix weights;
    Vector bias;
    Activation* activation;
    double lr;

    // Values saved from forward pass
    Vector input_cache;
    Vector z_cache;
    Vector output_cache;

public:
    Layer(size_t input_dim, size_t output_dim, Activation* act, double learning_rate = 0.01);

    Vector forward(const Vector& x);
    Vector backward(const Vector& dL_dy);
};
