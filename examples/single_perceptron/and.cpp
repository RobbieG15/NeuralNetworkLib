#include "perceptron.hpp"
#include "activation_funcs/sigmoid.hpp"
#include <iostream>
#include <vector>

int main() {
    Sigmoid act; 
    Perceptron p(2, &act, 0.001);

    std::vector<std::pair<Vector, double>> data = {
        { Vector{0.0, 0.0}, 0.0 },
        { Vector{0.0, 1.0}, 0.0 },
        { Vector{1.0, 0.0}, 0.0 },
        { Vector{1.0, 1.0}, 1.0 }
    };

    // Train
    for (int epoch = 0; epoch < 10000000; ++epoch) {
        for (auto& [x, target] : data) {
            p.train(x, target);
        }
    }

    // Test
    for (auto& [x, target] : data) {
        std::cout << x[0] << " AND " << x[1]
                  << " = " << p.forward(x)
                  << " (target " << target << ")\n";
    }
}
