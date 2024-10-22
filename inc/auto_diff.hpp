#include <iostream>
#include <cmath>
#include <numbers>
#include <Eigen/Dense>

namespace AutoDiff
{
    struct Expression
    {
        bool is_vector; // is the expression result a scalar or vector?
        double scalar_value;
        Eigen::RowVectorXd value;
        virtual void evaluate() = 0;
        virtual void derive(const Eigen::RowVectorXd &seed) = 0;
        friend std::ostream& operator<<(std::ostream& os, Expression &expr);
    };

    struct Variable : public Expression
    {
        Eigen::RowVectorXd partial;
        double partial_scalar;

        Variable(double &_scalar_value) {
            scalar_value = _scalar_value;
            partial_scalar = 0.0;
            is_vector = false;
        }

        Variable(Eigen::RowVectorXd &_value) {
            value = _value;
            partial = Eigen::RowVectorXd::Zero(_value.cols());
            is_vector = true;
        }

        void evaluate() { 
#ifdef AUTODIFF_DEBUG
            std::cout << "eval Var" << *this << std::endl;
#endif
        }
        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "partial update [seed]" << seed << std::endl;
#endif
            if(is_vector) {
                partial = partial + seed;
            }else{
                Eigen::RowVectorXd ones = Eigen::RowVectorXd::Ones(seed.cols());
                partial_scalar += (ones * seed.transpose());
            } 
        }
        friend std::ostream& operator<<(std::ostream& os, Variable &expr);
    };

    struct LinProd : public Expression
    {
        Expression *expr;
        Eigen::RowVectorXd vec;

        LinProd(Expression *_expr, const Eigen::RowVectorXd &_vec) : expr{_expr}, vec{_vec} {
            is_vector = false;
        }

        void evaluate() {
#ifdef AUTODIFF_DEBUG
        std::cout << "enter eval LinProd\n";
#endif
            expr->evaluate();

            assert(expr->value.rows() == vec.rows());
            scalar_value = expr->value * vec.transpose();

#ifdef AUTODIFF_DEBUG
            std::cout << "eval LinProd: " << *this << std::endl;
            std::cout << "(expr:  " << expr << " vec: " << vec << ")" << std::endl;
#endif
        }

        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive LinProd: [seed]" << seed << std::endl;
#endif
            if(expr->is_vector) {
                expr->derive(vec.array() * seed.array());
            }else{
                expr->derive(expr->scalar_value * seed.array());
            }
        }
    };

    struct QuadProd : public Expression 
    {
        Expression *expr;
        Eigen::MatrixXd A;

        QuadProd (Expression *_expr, const Eigen::MatrixXd &_A) : expr{_expr}, A{_A} {
            is_vector = false;
        }

        void evaluate() {
#ifdef AUTODIFF_DEBUG
        std::cout << "enter eval QuadProd\n";
#endif
            expr->evaluate();

            assert(expr->value.cols() == A.rows());
            scalar_value = expr->value * A * expr->value.transpose();

#ifdef AUTODIFF_DEBUG
            std::cout << "eval QuadProd: " << *this << std::endl;
            std::cout << "(expr:  " << expr << " A: " << A << ")" << std::endl;
#endif
        }

        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive QuadProd: [seed]" << seed << std::endl;
#endif
            if(expr->is_vector){
                Eigen::RowVectorXd grad = 2 * (A * expr->value.transpose()).transpose();
                expr->derive(grad.array() * seed.array());
            }else{
                expr->derive(expr->scalar_value * seed.array());
            }
        }
    };

    struct Pow : public Expression
    {
        Expression *expr;
        double exp;
        Pow(Expression *_expr, double exp) : expr{_expr}, exp{exp} {
            is_vector = true;
        }
        void evaluate() {
#ifdef AUTODIFF_DEBUG
        std::cout << "enter eval Pow\n";
#endif
            expr->evaluate();
            if(expr->is_vector) {
                value = expr->value.array().pow(-0.5);
                is_vector = true;
            } else {
                scalar_value = std::pow(expr->scalar_value,-0.5);
                is_vector = false;
            }
#ifdef AUTODIFF_DEBUG
            std::cout << "eval Pow: " << *this << std::endl;
            std::cout << "(expr:  " << expr << " exp = " << exp << ")" << std::endl;
#endif

        }
        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive Pow: [seed]" << seed << std::endl;
#endif
            if(expr->is_vector){
                Eigen::RowVectorXd grad = expr->value.array().pow(-1.5) * -0.5;
                expr->derive(grad.array() * seed.array());
            }else{
                expr->derive(expr->scalar_value * seed.array());
            }
        }
    };

    struct ElemProd : public Expression
    {
        Expression *a, *b;
        ElemProd(Expression *a, Expression *b) : a(a), b(b) {
            is_vector = true;
        }
        void evaluate() {
#ifdef AUTODIFF_DEBUG
        std::cout << "enter eval ElemProd\n";
#endif
            a->evaluate();
            b->evaluate();

            if (a->is_vector && b->is_vector) {
                value = a->value.array() * b->value.array();  

            } else if (!a->is_vector && b->is_vector) {
                value = b->value * a->scalar_value;  

            } else if (a->is_vector && !b->is_vector) {
                value = a->value * b->scalar_value;  
            }else{
                scalar_value = a->scalar_value * b->scalar_value;
                is_vector = false;
            }

#ifdef AUTODIFF_DEBUG
            std::cout << "eval ElemProd: " << *this << std::endl;
            std::cout << "(expr A:  " << a << " expr B: " << b << ")" << std::endl;
#endif
        }

        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive ElemProd: [seed]" << seed << std::endl;
#endif
            if(b->is_vector) a->derive(b->value.array() * seed.array());
            else a->derive(b->scalar_value * seed.array());

            if(a->is_vector) b->derive(a->value.array() * seed.array());
            else b->derive(a->scalar_value * seed.array());
        }
    };

    struct Sub : public Expression
    {
        Expression *a;
        double b;
        Sub(Expression *a, double b) : a(a), b(b) {
            is_vector = a->is_vector;
        }
        void evaluate() {
#ifdef AUTODIFF_DEBUG
            std::cout << "enter eval Sub\n";
#endif
            a->evaluate();
            if (a->is_vector) {
                value = a->value.array() - b;
            } else {
                scalar_value = a->scalar_value - b;
            }
#ifdef AUTODIFF_DEBUG
            std::cout << "eval Sub: " << *this << std::endl;
            std::cout << "(expr: " << a << " b: " << b << ")" << std::endl;
#endif
        }
        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive Sub: [seed]" << seed << std::endl;
#endif
            a->derive(seed); 
        }
    };

    struct Div : public Expression
    {
        Expression *numerator, *denominator;
        Div(Expression *numerator, Expression *denominator) : numerator(numerator), denominator(denominator) {
            is_vector = numerator->is_vector;
        }
        void evaluate() {
#ifdef AUTODIFF_DEBUG
            std::cout << "enter eval Div\n";
#endif
            numerator->evaluate();
            denominator->evaluate();
            if (is_vector) {
                value = numerator->value.array() / denominator->value.array();
            } else {
                scalar_value = numerator->scalar_value / denominator->scalar_value;
            }
#ifdef AUTODIFF_DEBUG
            std::cout << "eval Div: " << *this << std::endl;
            std::cout << "(numerator: " << numerator << " denominator: " << denominator << ")" << std::endl;
#endif
        }
        void derive(const Eigen::RowVectorXd &seed) {
#ifdef AUTODIFF_DEBUG
            std::cout << "derive Div: [seed]" << seed << std::endl;
#endif
            if (is_vector)
            {
                numerator->derive(seed.array() / denominator->value.array());
                denominator->derive(-seed.array() * numerator->value.array() / denominator->value.array().square());
            }
            else
            {
                numerator->derive(seed.array() / denominator->scalar_value);
                denominator->derive(-seed.array() * numerator->scalar_value / std::pow(denominator->scalar_value, 2));
            }
        }
    };
}