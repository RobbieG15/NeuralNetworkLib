#include "layer.hpp"
#include <random>

Layer::Layer(size_t input_dim, size_t output_dim, Activation* act, double learning_rate)
    : weights(output_dim, input_dim),
      bias(output_dim),
      activation(act),
      lr(learning_rate),
      input_cache(input_dim),
      z_cache(output_dim),
      output_cache(output_dim)
{
    // Random init weights (small values)
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (size_t i = 0; i < weights.numRows(); ++i) {
        for (size_t j = 0; j < weights.numCols(); ++j) {
            weights(i, j) = dist(gen);
        }
        bias[i] = 0.0;
    }
}

Vector Layer::forward(const Vector& x) {
    input_cache = x;
    z_cache = weights.multiply(x) + bias;
    output_cache = Vector(z_cache.size());
    for (size_t i = 0; i < z_cache.size(); i++) {
        output_cache[i] = activation->activate(z_cache[i]);
    }
    return output_cache;
}

Vector Layer::backward(const Vector& dL_dy) {
    // f'(z)
    Vector d_act = Vector(weights.numRows());
    for (size_t i = 0; i< weights.numRows(); i++) {
        d_act[i] = activation->derivative(z_cache[i]);
    }
    Vector dL_dz = dL_dy.hadamard(d_act);

    // Gradients
    Matrix dL_dW = outer(dL_dz, input_cache);
    Vector dL_db = dL_dz;

    // Backprop gradient to input
    Vector dL_dx = weights.transpose().multiply(dL_dz);

    // Update
    weights = weights - (dL_dW * lr);
    bias = bias - (dL_db * lr);

    return dL_dx;
}
