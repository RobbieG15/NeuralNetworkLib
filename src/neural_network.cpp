#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<Layer>& layers) : layers(layers) {};

Vector NeuralNetwork::forward(const Vector& x) {
    Vector out = x;
    for (size_t i = 0; i < layers.size(); ++i) {
        out = layers[i].forward(out);
    }
    return out;
}

void NeuralNetwork::backward(const Vector& dL_dy) {
    Vector grad = dL_dy;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = it->backward(grad);
    }
}