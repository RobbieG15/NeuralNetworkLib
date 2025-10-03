#pragma once
#include "vector.hpp"
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <iostream>

class Matrix {
private:
    size_t rows;
    size_t cols;
    Vector data;

public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, std::initializer_list<double> init);

    // Accessors
    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    // Indexing
    double& operator()(size_t r, size_t c);
    const double& operator()(size_t r, size_t c) const;

    // Basic operations
    Matrix transpose() const;
    Vector multiply(const Vector& v) const;
    Matrix hadamard(const Matrix& m) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator* (const double other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;

    // Utility
    void print() const;
};

Matrix outer(const Vector& a, const Vector& b);