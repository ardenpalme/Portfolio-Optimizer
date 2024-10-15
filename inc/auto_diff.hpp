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
    };

    struct Variable : public Expression
    {
        Eigen::RowVectorXd partial;
        Variable(const Eigen::RowVectorXd &_value) {
            value = _value;
            partial = Eigen::RowVectorXd::Zero(_value.cols());
            is_vector = true;
        }
        void evaluate() { }
        void derive(const Eigen::RowVectorXd &seed) {
            partial = partial + seed;
        }
    };

    struct LinearProd : public Expression
    {
        Expression *expr;
        Eigen::RowVectorXd vec;

        LinearProd(Expression *_expr, const Eigen::RowVectorXd &_vec) : expr{_expr}, vec{_vec} {
            is_vector = false;
        }

        void evaluate() {
            expr->evaluate();
            scalar_value = expr->value.dot(vec);
        }

        void derive(const Eigen::RowVectorXd &seed) {
            if(expr->is_vector) expr->derive(vec.array() * seed.array());
            else expr->derive(expr->scalar_value * seed.array());
        }
    };

    struct VecT_Matrix_Vec : public Expression {
        Expression *expr;
        Eigen::MatrixXd A;

        VecT_Matrix_Vec(Expression *_expr, const Eigen::MatrixXd &_A) : expr{_expr}, A{_A} {
            is_vector = false;
        }

        void evaluate() {
            expr->evaluate();
            scalar_value = expr->value * A * expr->value.transpose();
        }

        void derive(const Eigen::RowVectorXd &seed) {
            if(expr->is_vector){
                Eigen::RowVectorXd grad = 2 * (A * expr->value.transpose()).transpose();
                expr->derive(grad.array() * seed.array());
            }else{
                expr->derive(expr->scalar_value * seed.array());
            }
        }
    };

    struct Power : public Expression
    {
        Expression *expr;
        double exp;
        Power(Expression *_expr, double exp) : expr{_expr}, exp{exp} {
            is_vector = true;
        }
        void evaluate() {
            expr->evaluate();
            if(expr->is_vector) {
                value = expr->value.array().pow(-0.5);
                is_vector = true;

            } else {
                scalar_value = std::pow(expr->scalar_value,-0.5);
                is_vector = false;
            }

        }
        void derive(const Eigen::RowVectorXd &seed) {
            if(expr->is_vector){
                Eigen::RowVectorXd grad = expr->value.array().pow(-1.5) * -0.5;
                expr->derive(grad.array() * seed.array());
            }else{
                expr->derive(expr->scalar_value * seed.array());
            }
        }
    };

    struct Multiply : public Expression
    {
        Expression *a, *b;
        Multiply(Expression *a, Expression *b) : a(a), b(b) {
            is_vector = true;
        }
        void evaluate() {
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
        }

        void derive(const Eigen::RowVectorXd &seed) {
            if(b->is_vector) a->derive(b->value.array() * seed.array());
            else a->derive(b->scalar_value * seed.array());

            if(a->is_vector) b->derive(a->value.array() * seed.array());
            else b->derive(a->scalar_value * seed.array());
        }
    };
}