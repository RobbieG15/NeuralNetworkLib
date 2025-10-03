#include <iostream>
#include <vector>
#include <cmath>
#include "vector.hpp"
#include "layer.hpp"
#include "neural_network.hpp"
#include "activation_funcs/sigmoid.hpp"
#include "activation_funcs/tanh.hpp"
#include "losses.hpp"

int main() {
    Sigmoid sigmoid;
    Tanh tanh;

    Layer hidden(2, 2, &tanh, 0.25);
    Layer output(2, 1, &sigmoid, 0.25);

    std::vector<Layer> layers = {hidden, output};
    NeuralNetwork network(layers);

    std::vector<Vector> inputs = {
        Vector{0, 0}, Vector{0, 1}, Vector{1, 0}, Vector{1, 1}
    };
    std::vector<Vector> targets = {
        Vector{0}, Vector{1}, Vector{1}, Vector{0}
    };

    for (int epoch = 0; epoch < 50000; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Vector output = network.forward(inputs[i]);
            Vector loss_grad = mse_loss_gradient(output, targets[i]);
            network.backward(loss_grad);

            epoch_loss += mse_loss(output, targets[i]);
        }

        if (epoch % 500 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector output = network.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] ";
        std::cout << "Output: " << output[0] << ", Target: " << targets[i][0] << std::endl;
    }

    return 0;
}
