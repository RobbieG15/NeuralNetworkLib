#pragma once
#include "vector.hpp"

double mse_loss(const Vector& predicted, const Vector& target) {
    if (predicted.size() != target.size()) throw std::runtime_error("Vector-Vector size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double difference = predicted[i] - target[i];
        sum += difference * difference;
    } 
    return sum;
}

Vector mse_loss_gradient(const Vector& predicted, const Vector& target) {
    if (predicted.size() != target.size()) throw std::runtime_error("Vector-Vector size mismatch");
    Vector grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        grad[i] = 2.0 * (predicted[i] - target[i]) / predicted.size();
    }
    return grad;
}