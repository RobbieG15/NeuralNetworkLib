#pragma once

class Activation {
public:
    virtual ~Activation() = default;

    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
};
