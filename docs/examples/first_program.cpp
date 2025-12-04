#include "../../autograd.hpp"

int main() {
    // 创建变量
    auto x = make_var(3.0);
    auto w = make_var(2.0);
    auto b = make_var(1.0);
    
    // 构建计算图：y = w * x + b
    auto y = add(mul(w, x), b);
    
    // 前向计算
    y->calc();
    std::cout << "y = " << y->item() << std::endl;  // 输出: y = 7
    
    // 反向传播
    y->backward();
    
    // 查看梯度
    std::cout << "∂y/∂w = " << w->grad_item() << std::endl;  // 输出: 3
    std::cout << "∂y/∂x = " << x->grad_item() << std::endl;  // 输出: 2
    std::cout << "∂y/∂b = " << b->grad_item() << std::endl;  // 输出: 1
    
    return 0;
}