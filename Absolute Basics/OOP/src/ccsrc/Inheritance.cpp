#include "Inheritance.h"
#include <iostream>

Derived::Derived(int value, int extraValue) : Base(value), extraValue(extraValue) {}

void Derived::showExtraValue() {
    std::cout << "Extra Value: " << extraValue << std::endl;
}

