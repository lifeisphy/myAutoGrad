#include "../autograd.hpp"
#include <iostream>

int main(){
    VarPtr a = make_var({1.0, 2.0, 3.0, 4.0,5.0,6.0}, {2,3});
    // VarPtr b = make_var({4.0, 5.0, 6.0, 7.0,8.0,9.0}, true, {3,2});
    VarPtr b = make_var({4,5,6,7,8,9}, {2,3});
    // VarPtr d = mul(a,b,1,0);
    VarPtr d = mul_elementwise(a,b);

    auto s = sum(d);
    s->backward();
    d->print();
    a->print();
    b->print();
    return 0;
}