#ifndef BASE_H
#define BASE_H

class Base {
public:
    Base(int value);
    void showValue();
    void publicMethod();
    virtual void virtualMethod(); // Virtual function for overriding

protected:
    void protectedMethod();

private:
    void privateMethod();
    int value;
};

#endif // BASE_H
