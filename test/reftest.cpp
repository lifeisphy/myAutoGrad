#include "../autograd.hpp"
int main(){
    auto a = make_input({1.0,2.0,}, {2});
    auto d = make_param({10.0,20.0}, {2});
    auto b = slice(d, {0});
    auto b2 = slice(d, {1});
    auto c = add(mul_elementwise(b,make_input(2)),b2);
    c->zero_grad_recursive();
    c->calc();
    c->backward();
    for(auto v: {a,b,b2,c,d}){
        v->print(std::cout, true);  
    }
}

