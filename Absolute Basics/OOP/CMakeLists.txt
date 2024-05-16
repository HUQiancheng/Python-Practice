cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(LearnOOP)
set(CMAKE_CXX_STANDARD 11)

# Add source files
set(SOURCES
    src/ccsrc/Base.cpp
    src/ccsrc/Inheritance.cpp
    src/ccsrc/Polymorphism.cpp
)

# Add header files directory
include_directories(src/ccsrc)

# Create the executable
add_executable(LearnOOP learnoop.cpp src/ccsrc/Base.cpp src/ccsrc/Inheritance.cpp src/ccsrc/Polymorphism.cpp)
