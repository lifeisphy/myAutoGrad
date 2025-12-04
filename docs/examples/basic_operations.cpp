#include "../../autograd.hpp"
#include <iostream>

int main() {
    // 基础运算
    auto a = make_var(2.0);
    auto b = make_var(3.0);

    auto sum = add(a, b);        // 2 + 3 = 5
    auto diff = sub(a, b);       // 2 - 3 = -1
    auto prod = mul(a, b);       // 2 * 3 = 6
    auto power = pow_elementwise(a, 3);  // 2^3 = 8

    // 使用操作符
    auto sum2 = a + b;          // 等同于 add(a, b)
    auto diff2 = a - b;         // 等同于 sub(a, b)
    auto prod2 = a * b;         // 等同于 mul(a, b)

    // 向量运算
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2 = {4.0, 5.0, 6.0};

    auto vec1 = make_var(data1);
    auto vec2 = make_var(data2);

    auto vec_sum = add(vec1, vec2);        // [5, 7, 9]
    auto vec_prod = mul_elementwise(vec1, vec2);  // [4, 10, 18]
    auto vec_total = sum({vec_sum});         // 21

    // 广播机制
    auto scalar = make_var(2.0);
    auto vector = make_var(std::vector<double>{1.0, 2.0, 3.0});

    auto broadcast_mul = mul(vector, scalar);  // [2, 4, 6]

    // 计算并打印结果
    sum->calc();
    diff->calc();
    prod->calc();
    power->calc();
    
    std::cout << "Basic operations:" << std::endl;
    std::cout << "2 + 3 = " << sum->item() << std::endl;
    std::cout << "2 - 3 = " << diff->item() << std::endl;
    std::cout << "2 * 3 = " << prod->item() << std::endl;
    std::cout << "2^3 = " << power->item() << std::endl;

    vec_sum->calc();
    vec_prod->calc();
    vec_total->calc();
    
    std::cout << "\nVector operations:" << std::endl;
    std::cout << "[1,2,3] + [4,5,6] = ";
    vec_sum->print(std::cout);
    std::cout << std::endl;
    std::cout << "[1,2,3] * [4,5,6] = ";
    vec_prod->print(std::cout);
    std::cout << std::endl;
    std::cout << "sum([5,7,9]) = " << vec_total->item() << std::endl;

    broadcast_mul->calc();
    std::cout << "\nBroadcasting:" << std::endl;
    std::cout << "[1,2,3] * 2 = ";
    broadcast_mul->print(std::cout);
    
    return 0;
}