#include <iostream>
#include <cmath>
#include <numbers>

struct Expression
{
    float value;
    virtual void evaluate() = 0;
    virtual void derive(float seed) = 0;
};

struct Variable : public Expression
{
    float partial;
    Variable(float value)
    {
        this->value = value;
        partial = 0.0f;
    }
    void evaluate() {}
    void derive(float seed)
    {
        partial += seed;
    }
};

struct Plus : public Expression
{
    Expression *a, *b;
    Plus(Expression *a, Expression *b) : a(a), b(b) {}
    void evaluate()
    {
        a->evaluate();
        b->evaluate();
        value = a->value + b->value;
    }
    void derive(float seed)
    {
        a->derive(seed);
        b->derive(seed);
    }
};

struct Multiply : public Expression
{
    Expression *a, *b;
    Multiply(Expression *a, Expression *b) : a(a), b(b) {}
    void evaluate()
    {
        a->evaluate();
        b->evaluate();
        value = a->value * b->value;
    }
    void derive(float seed)
    {
        a->derive(b->value * seed);
        b->derive(a->value * seed);
    }
};

struct Sine : public Expression
{
    Expression *a;
    Sine(Expression *a) : a(a) {}
    void evaluate() {
        a->evaluate();
        value = std::sin(a->value);
    }
    void derive(float seed) {
        a->derive(std::cos(a->value) * seed);
    }
};

/*
int main(int argc, char *argv[])
{
    // f(x1,x2) = x1 * x2 + sin(x1)
    Variable x1(3.1415926), x2(1);
    Multiply m1(&x1, &x2);
    Sine s1(&x1);
    Plus z(&m1, &s1);

    z.evaluate();
    std::cout << "f(x1 = " << x1.value << ", x2 = " << x2.value << ") = " << z.value << std::endl;

    z.derive(1);
    std::cout << "∂f/∂x1 = " << x1.partial << std::endl
              << "∂f/∂x2 = " << x2.partial << std::endl;

    return 0;
}
*/