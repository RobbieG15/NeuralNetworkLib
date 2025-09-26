find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ../include)

add_executable(run_tests test_vector.cpp ../src/vector.cpp)
target_link_libraries(run_tests GTest::gtest_main pthread)
add_test(NAME VectorTests COMMAND run_tests)
