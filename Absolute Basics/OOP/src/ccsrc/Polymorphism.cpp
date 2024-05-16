#include "Polymorphism.h"
#include <iostream>

Polymorphism::Polymorphism(int value) : Base(value) {}

void Polymorphism::virtualMethod() {
    std::cout << "Polymorphic Derived Class Virtual Method Called" << std::endl;
}
