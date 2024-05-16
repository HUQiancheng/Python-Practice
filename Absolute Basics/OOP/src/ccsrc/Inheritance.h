#ifndef INHERITANCE_H
#define INHERITANCE_H

#include "Base.h"

class Derived : public Base {
public:
    Derived(int value, int extraValue);
    void showExtraValue();

private:
    int extraValue;
};

#endif // INHERITANCE_H
