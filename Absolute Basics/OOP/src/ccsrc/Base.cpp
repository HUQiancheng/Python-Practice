#include "Base.h"
#include <iostream>

Base::Base(int value) : value(value) {}

void Base::showValue() {
    std::cout << "Value: " << value << std::endl;
}

void Base::publicMethod() {
    std::cout << "Public Method Called From Base" << std::endl;
    privateMethod(); // Demonstrate calling a private method
}

void Base::virtualMethod() {
    std::cout << "Base Virtual Method Called From Base" << std::endl;
}

void Base::protectedMethod() {
    std::cout << "Protected Method Called From Base" << std::endl;
}

void Base::privateMethod() {
    std::cout << "Private Method Called From Base" << std::endl;
}
