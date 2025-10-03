#pragma once
#include "vector.hpp"
#include "layer.hpp"

class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    NeuralNetwork(const std::vector<Layer>& layers);

    Vector forward(const Vector& x);
    void backward(const Vector& dL_dy);
};