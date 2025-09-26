#include <gtest/gtest.h>
#include "vector.hpp"
#include <stdexcept>
#include <cmath>

const double EPS = 1e-9;

TEST(VectorConstructorTest, DefaultConstructor) {
    Vector v(3);
    EXPECT_EQ(v.size(), 3);
    for (size_t i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(v[i], 0.0);
}

TEST(VectorConstructorTest, InitListConstructor) {
    Vector v{1.0, 2.0, 3.0};
    EXPECT_EQ(v.size(), 3);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

TEST(VectorAccessTest, ElementAccess) {
    Vector v{1.0, 2.0};
    v[0] = 5.0;
    EXPECT_DOUBLE_EQ(v[0], 5.0);
    EXPECT_THROW(v[2], std::out_of_range);
}

TEST(VectorFillTest, FillValues) {
    Vector v(3);
    v.fill(7.0);
    for (size_t i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(v[i], 7.0);
}

TEST(VectorArithmeticTest, Addition) {
    Vector a{1,2,3}, b{4,5,6};
    Vector c = a + b;
    EXPECT_DOUBLE_EQ(c[0], 5.0);
    EXPECT_DOUBLE_EQ(c[1], 7.0);
    EXPECT_DOUBLE_EQ(c[2], 9.0);
    Vector d{1,2};
    EXPECT_THROW(a + d, std::runtime_error);
}

TEST(VectorArithmeticTest, Subtraction) {
    Vector a{5,6,7}, b{1,2,3};
    Vector c = a - b;
    EXPECT_DOUBLE_EQ(c[0], 4.0);
    EXPECT_DOUBLE_EQ(c[1], 4.0);
    EXPECT_DOUBLE_EQ(c[2], 4.0);
    Vector d{1,2};
    EXPECT_THROW(a - d, std::runtime_error);
}

TEST(VectorArithmeticTest, UnaryMinus) {
    Vector v{1,-2,3};
    Vector u = -v;
    EXPECT_DOUBLE_EQ(u[0], -1.0);
    EXPECT_DOUBLE_EQ(u[1], 2.0);
    EXPECT_DOUBLE_EQ(u[2], -3.0);
}

TEST(VectorArithmeticTest, ScalarMultiplicationDivision) {
    Vector v{1,2,3};
    Vector u = v * 2.0;
    Vector w = v / 2.0;
    EXPECT_DOUBLE_EQ(u[0], 2.0);
    EXPECT_DOUBLE_EQ(u[1], 4.0);
    EXPECT_DOUBLE_EQ(u[2], 6.0);
    EXPECT_DOUBLE_EQ(w[0], 0.5);
    EXPECT_DOUBLE_EQ(w[1], 1.0);
    EXPECT_DOUBLE_EQ(w[2], 1.5);
}

TEST(VectorProductTest, DotProduct) {
    Vector a{1,2,3}, b{4,5,6};
    double result = a.dot(b);
    EXPECT_DOUBLE_EQ(result, 32.0);
    Vector c{1,2};
    EXPECT_THROW(a.dot(c), std::runtime_error);
}

TEST(VectorProductTest, CrossProduct) {
    Vector a{1,0,0}, b{0,1,0};
    Vector c = a.cross(b);
    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
    EXPECT_DOUBLE_EQ(c[2], 1.0);
    Vector d{1,2};
    EXPECT_THROW(a.cross(d), std::runtime_error);
}

TEST(VectorNormTest, L2Norm) {
    Vector v{3,4};
    EXPECT_NEAR(v.l2norm(), 5.0, EPS);
}

TEST(VectorNormTest, L1Norm) {
    Vector v{1,-2,3};
    EXPECT_DOUBLE_EQ(v.l1norm(), 6.0);
}

TEST(VectorNormTest, LinfNorm) {
    Vector v{1,-5,3};
    EXPECT_DOUBLE_EQ(v.linfnorm(), 5.0);
}

TEST(VectorNormTest, UnitVector) {
    Vector v{3,4};
    Vector u = v.unit();
    EXPECT_NEAR(u.l2norm(), 1.0, EPS);
    EXPECT_NEAR(u[0], 3.0/5.0, EPS);
    EXPECT_NEAR(u[1], 4.0/5.0, EPS);
}

TEST(VectorDistanceTest, DistanceBetweenVectors) {
    Vector a{1,2,3}, b{4,5,6};
    double dist = a.distance(b);
    EXPECT_NEAR(dist, std::sqrt(27.0), EPS);
}

