# Find all cpp files recursively in this directory and subdirectories
file(GLOB_RECURSE EXAMPLES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

# Loop over each file and make an executable
foreach(example_file ${EXAMPLES})
    # Get filename without extension
    get_filename_component(example_name ${example_file} NAME_WE)

    add_executable(${example_name} ${CMAKE_CURRENT_SOURCE_DIR}/${example_file})
    target_link_libraries(${example_name} PRIVATE VectorLib PerceptronLib)
    target_link_libraries(${example_name} PRIVATE VectorLib NeuralNetworkLib)
    target_link_libraries(${example_name} PRIVATE VectorLib LayerLib)
    target_link_libraries(${example_name} PRIVATE VectorLib MatrixLib)
    target_include_directories(${example_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()
