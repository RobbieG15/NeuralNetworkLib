#include <iostream>
#include <vector>
#include <cmath>
#include "vector.hpp"
#include "layer.hpp"
#include "neural_network.hpp"
#include "activation_funcs/sigmoid.hpp"
#include "activation_funcs/relu.hpp"
#include "losses.hpp"

// Variance in results can occur because of initialization of weights and other factors
// Running this training a few times should be enough to get good results
int main() {
    Sigmoid sigmoid;
    ReLU relu;

    // N hidden neurons is sufficient to solve N-bit parity in a fully connected network
    // 3 input features -> 3 hidden neurons -> 1 output
    Layer hidden(3, 3, &relu, 0.25);
    Layer output(3, 1, &sigmoid, 0.25);

    std::vector<Layer> layers = {hidden, output};
    NeuralNetwork network(layers);

    // All 3-bit input combinations
    std::vector<Vector> inputs = {
        Vector{0, 0, 0}, Vector{0, 0, 1}, Vector{0, 1, 0}, Vector{0, 1, 1},
        Vector{1, 0, 0}, Vector{1, 0, 1}, Vector{1, 1, 0}, Vector{1, 1, 1}
    };

    // Parity targets: 1 if odd number of 1s, 0 if even
    std::vector<Vector> targets = {
        Vector{0}, Vector{1}, Vector{1}, Vector{0},
        Vector{1}, Vector{0}, Vector{0}, Vector{1}
    };

    // Training loop
    for (int epoch = 0; epoch < 50000; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Vector out = network.forward(inputs[i]);
            Vector loss_grad = mse_loss_gradient(out, targets[i]);
            network.backward(loss_grad);

            epoch_loss += mse_loss(out, targets[i]);
        }

        if (epoch % 5000 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
    }

    // Print results
    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector out = network.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << ", " << inputs[i][2] << "] ";
        std::cout << "Output: " << out[0] << ", Target: " << targets[i][0] << std::endl;
    }

    return 0;
}
