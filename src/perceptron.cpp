#include "perceptron.hpp"
#include <cstdlib>
#include <cmath>

Perceptron::Perceptron(size_t input_dim, Activation* act, double learning_rate)
    : weights(input_dim), bias(0.0), lr(learning_rate), activation(act) {
    // Randomly initialize weight vector
    for (size_t i = 0; i < input_dim; ++i) {
        weights[i] = (double(rand()) / RAND_MAX) * 0.01;
    }
}

double Perceptron::forward(const Vector& x) const {
    double z = weights.dot(x) + bias;
    return activation->activate(z);
}

void Perceptron::train(const Vector& x, double target) {
    double z = weights.dot(x) + bias;
    double y_hat = activation->activate(z);

    double error = target - y_hat;
    double grad = error * activation->derivative(z);

    // Update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += lr * grad * x[i];
    }

    // Update bias
    bias += lr * grad;
}
