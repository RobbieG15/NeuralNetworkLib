#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

class Vector {
public:
    // Constructors
    explicit Vector(size_t n);
    Vector(std::initializer_list<double> init);

    // Size
    size_t size() const;

    // Element access
    double& operator[](size_t i);
    const double& operator[](size_t i) const;

    // Fill
    void fill(double val);

    // Basic arithmetic
    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator-() const;
    Vector operator*(double scalar) const;
    Vector operator/(double scalar) const;
    double dot(const Vector& other) const;
    Vector cross(const Vector& other) const;
    Vector hadamard(const Vector& other) const;

    // Norms and Distances
    double l2norm() const;
    double l1norm() const;
    double linfnorm() const;
    Vector unit() const;
    double distance(const Vector& other) const;


    // Utilities
    void print() const;

private:
    std::vector<double> data_;
};
