#include "../../autograd.hpp"
#include <iostream>

int main() {
    // 使用函数创建变量
    auto x = make_input(2.0);
    auto w = make_param(0.5);
    auto b = make_param(0.1);
    
    // 使用函数构建计算图
    auto wx = mul(w, x);
    auto y = add(wx, b);
    
    // 前向计算
    y->calc();
    std::cout << "y = " << y->item() << std::endl;
    
    // 反向传播
    y->backward();
    std::cout << "dw = " << w->grad()[0] << std::endl;
    // x是输入变量，默认没有梯度
    if (x->has_grad() && x->grad().size() > 0) {
        std::cout << "dx = " << x->grad()[0] << std::endl;
    } else {
        std::cout << "dx = (no gradient for input)" << std::endl;
    }
    
    return 0;
}