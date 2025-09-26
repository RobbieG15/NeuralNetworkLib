#include "vector.hpp"
#include <cmath>

// Constructors
Vector::Vector(size_t n) : data_(n, 0.0) {}

Vector::Vector(std::initializer_list<double> init) : data_(init) {}

// Size
size_t Vector::size() const { return data_.size(); }

// Element access
double& Vector::operator[](size_t i) {
    if (i >= data_.size()) throw std::out_of_range("Index out of range");
    return data_[i];
}

const double& Vector::operator[](size_t i) const {
    if (i >= data_.size()) throw std::out_of_range("Index out of range");
    return data_[i];
}

// Fill
void Vector::fill(double val) {
    for (auto& x : data_) x = val;
}

// Arithmetic
Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch in addition");
    Vector result(size());
    for (size_t i = 0; i < size(); i++) result[i] = data_[i] + other[i];
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch in subtraction");
    Vector result(size());
    for (size_t i = 0; i < size(); i++) result[i] = data_[i] - other[i];
    return result;
}

Vector Vector::operator-() const {
    Vector result(size());
    for (size_t i=0; i < size(); i++) result[i] = data_[i] * -1;
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(size());
    for (size_t i = 0; i < size(); i++) result[i] = data_[i] * scalar;
    return result;
}

Vector Vector::operator/(double scalar) const {
    Vector result(size());
    for (size_t i=0; i < size(); i++) result[i] = data_[i] / scalar;
    return result;
}

double Vector::dot(const Vector& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch in dot product");
    double sum = 0.0;
    for (size_t i = 0; i < size(); i++) sum += data_[i] * other[i];
    return sum;
}

Vector Vector::cross(const Vector& other) const {
    if (size() != other.size() || size() != 3) throw std::runtime_error("Cross product is only defined for 3-D vectors");
    Vector result(size());
    result[0] = data_[1] * other[2] - data_[2] * other[1];
    result[1] = data_[2] * other[0] - data_[0] * other[2];
    result[2] = data_[0] * other[1] - data_[1] * other[0];
    return result;
}

// Norms and Distances
double Vector::l2norm() const {
    return std::sqrt(dot(*this));
}

double Vector::l1norm() const {
    double abs_sum = 0.0;
    for (auto i=0; i < size(); i++) abs_sum += std::abs(data_[i]);
    return abs_sum;
}

double Vector::linfnorm() const {
    double max = std::abs(data_[0]);
    for (auto i=1; i < size(); i++) {
        double i_abs = std::abs(data_[i]);
        if (i_abs > max) max = i_abs; 
    }
    return max;
}

Vector Vector::unit() const {
    double l2_norm = l2norm();
    return *this / l2_norm;
}

double Vector::distance(const Vector& other) const {
    Vector difference = *this - other;
    return difference.l2norm();
}

// Print
void Vector::print() const {
    std::cout << "[";
    for (size_t i = 0; i < size(); i++) {
        std::cout << data_[i];
        if (i + 1 < size()) std::cout << ", ";
    }
    std::cout << "]\n";
}
