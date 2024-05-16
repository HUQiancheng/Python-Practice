#include "ccsrc.h"
#include <iostream>

int main() {
    Base base(10);
    std::cout << "Base class demonstration:" << std::endl;
    base.showValue();
    base.publicMethod();
    base.virtualMethod();

    std::cout << std::endl;

    Derived derived(20, 30);
    std::cout << "Derived class demonstration:" << std::endl;
    derived.showValue();
    derived.showExtraValue();
    derived.publicMethod();
    derived.virtualMethod();

    std::cout << std::endl;

    // Polymorphism demonstration using base class pointer
    Base* polyBase = new Polymorphism(40);
    std::cout << "Polymorphism demonstration:" << std::endl;
    polyBase->showValue();
    polyBase->publicMethod();
    polyBase->virtualMethod(); // Calls the overridden method in Polymorphism class
    delete polyBase;

    return 0;
}
