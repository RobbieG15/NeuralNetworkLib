#include "matrix.hpp"

// Constructors
Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}

Matrix::Matrix(size_t r, size_t c, std::initializer_list<double> init)
    : rows(r), cols(c), data(r * c) {
    if (init.size() != r * c) {
        throw std::invalid_argument("Initializer list size does not match matrix size");
    }
    size_t i = 0;
    for (auto val : init) {
        data[i++] = val;
    }
}

// Indexing
double& Matrix::operator()(size_t r, size_t c) {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[r * cols + c];
}

const double& Matrix::operator()(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[r * cols + c];
}

// Basic operations
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Vector Matrix::multiply(const Vector& v) const {
    if (cols != v.size()) {
        throw std::invalid_argument("Matrix-Vector dimension mismatch");
    }
    Vector result(rows);
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += (*this)(r, c) * v[c];
        }
        result[r] = sum;
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& m) const {
    if (cols != m.cols || rows != m.rows) {
        throw std::invalid_argument("Matrix-Matrix dimension mismatch");
    }
    Matrix result(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result(r, c) = (*this)(r, c) * m(r, c);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix-Matrix dimension mismatch");
    }
    Matrix result(rows, other.cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < other.cols; ++c) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(r, k) * other(k, c);
            }
            result(r, c) = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(const double other) const {
    Matrix result(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result(r, c) = (*this)(r, c) * other;
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix addition dimension mismatch");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix subtraction dimension mismatch");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Print
void Matrix::print() const {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            std::cout << (*this)(r, c) << " ";
        }
        std::cout << "\n";
    }
}

Matrix outer(const Vector& a, const Vector& b) {
    Matrix result(a.size(), b.size());
    for (size_t a_size = 0; a_size < a.size(); a_size++) {
        for (size_t b_size = 0; b_size < b.size(); b_size++) {
            result(a_size, b_size) = a[a_size] * b[b_size];
        }
    }
    return result;
}