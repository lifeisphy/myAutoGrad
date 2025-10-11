#include "../autograd.hpp"
#include <iostream>
using namespace std;
int main(){
    VarPtr a = make_var(std::vector<double>{1.0, 2.0, 3.0}, {3});
    VarPtr b = make_var(std::vector<double>{4.0, 5.0, 6.0}, {3});
    auto c = a - b;
    // auto d = a - b;
    auto e = make_var(2);
    auto d = e * c;
    auto f = c * e;
    auto s = sum( pow_elementwise(f,2));
    std::vector<VarPtr> vars = {a, b, c, d, e, f, s};
    for (const auto &var : vars)
    {
        var->print();
    }
    e->print();
    cout<<"==================="<<endl;
    for(int i =0; i < 100; i++){
        double lr = 0.01;
        s->zero_grad_recursive();
        s->calc();
        s->backward();
        a->update(lr);
        b->update(lr);
        std::cout<< "i:" << i << " s->item(): " << s->item() << std::endl;
    }
    for (const auto &var : vars)
    {
        var->print();
    }
    return 0;
}