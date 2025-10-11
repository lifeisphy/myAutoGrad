#include "../autograd.hpp"
using namespace std;
std::vector<double> vec(size_t size){
    std::vector<double> data(size);
    for(size_t i=0; i < size; i++){
        data[i] = i;
    }
    return data;
}
std::vector<double> vec_r(size_t size){
    std::vector<double> data(size);
    for(size_t i=0; i < size; i++){
        data[i] = rand() / double(RAND_MAX);
    }
    return data;
}
int main(){
    auto a = make_var(vec(4), {4});
    auto W = make_var(vec(4 * 2), {4, 2});
    auto c = mul(W,a,0,0);
    auto W0 = make_var(vec_r(4*2), {4, 2});
    auto e = sum(mul_elementwise(c,c));
    for(int i=0; i<30; i++){
        e->zero_grad_recursive();
        e->calc();
        cout<< e->item() <<endl;
        e->backward();
        a->update(0.1);
    }
    for(auto v: {a,W,c,e}){
        v->print();
    }
}