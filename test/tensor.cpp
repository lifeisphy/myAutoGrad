#include "../autograd.hpp"
#include <iostream>

int main(){
    VarPtr a = make_var({1.0, 2.0, 3.0, 4.0,5.0,6.0}, true, {2,3});
    VarPtr b = make_var({4,5,6,7,8,9}, true, {2,3});
    VarPtr c = tensor(a,b);
    auto s = sum(c);
    s->backward();
    c->print();
    a->print();
    b->print();
    return 0;
}