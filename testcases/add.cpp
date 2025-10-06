#include "../autograd.hpp"
#include <iostream>

int main(){
    VarPtr a = make_var(std::vector<double>{1.0, 2.0, 3.0}, true, {3},false);
    VarPtr b = make_var(std::vector<double>{4.0, 5.0, 6.0}, true, {3});
    auto c = a - b;
    // auto d = a - b;
    auto e = make_var(2, false, {}, false);
    auto d = e * c;
    auto f = c * e;
    auto s = sum(f);
    std::vector<VarPtr> vars = {a, b, c, d, e, f, s};
    for (const auto &var : vars)
    {
        var->print();
    }
    e->print();
    cout<<"==================="<<endl;
    for(int i =0; i < 100; i++){
        double lr = 0.001;
        s->recursive_zero_grad();
        s->forward();
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