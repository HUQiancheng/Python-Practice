#ifndef POLYMORPHISM_H
#define POLYMORPHISM_H
#include "Base.h"

class Polymorphism: public Base {

public:
    Polymorphism(int value);
    // Override virtual void virtualMethod() at Base class
    void virtualMethod() override;
};
#endif // POLYMORPHISM_H
