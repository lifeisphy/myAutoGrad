/**
 * C++ 自动微分框架主程序
 * 基础的自动微分实现，支持标量和向量运算
 */
#include <cstdlib> // for rand()
#include <ctime>   // for time()
#include <algorithm> // for std::transform
#include "../autograd.hpp"
std::vector<double> random_vector(size_t size, double min_val=0.0, double max_val=1.0) {
    std::vector<double> vec(size);
    for (size_t i = 0; i < size; ++i) {
        double random_val = min_val + static_cast<double>(rand()) / RAND_MAX * (max_val - min_val);
        vec[i] = random_val;
    }
    return vec;
}
std::vector<double> normal_distribution_vector(size_t size, double mean=0.0, double stddev=1.0) {
    std::vector<double> vec(size);
    for (size_t i = 0; i < size; ++i) {
        // 使用Box-Muller变换生成正态分布随机数
        double u1 = static_cast<double>(rand()) / RAND_MAX;
        double u2 = static_cast<double>(rand()) / RAND_MAX;
        double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        vec[i] = z0 * stddev + mean;
    }
    return vec;
}
size_t n_input = 3;
size_t n_output = 2;
void gen_xy(std::vector<double> &x, std::vector<double> &y, const std::vector<double> &w) {
    x = random_vector(n_input, 0.0, 10.0);
    for(size_t i =0; i < n_output; i++){
        for(size_t j = 0 ; j < n_input; j++){
            y[i] += w[i*n_input + j] * x[j];
        }
        // y[i] += normal_distribution_vector(1, 0.0, 0.3)[0]; // noise;
    }
    std::cout<< "x: ";
    for (const auto& val : x) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout<< "y: ";
    for (const auto& val : y) {        
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
int main() {
    // 简单神经网络示例
    std::cout << "\n5. 简单神经网络训练示例:" << std::endl;

    std::vector<double> w_real = normal_distribution_vector(n_input * n_output, 2.0, 0.1);
    auto x = make_input(std::vector<double>(n_input,0.0), {n_input});
    auto y = make_input(std::vector<double>(n_output,0.0), {n_output});
    auto w = make_param(std::vector<double>(n_input * n_output, 0.0) , {n_input, n_output});
    auto yhat = mul(w,x,0,0);
    auto loss = mse_loss(yhat, y);
    double learning_rate = 0.01;
    
    int ndata = 10;
    std::vector<std::vector<double>> x_data(ndata, std::vector<double>(n_input,0.0));
    std::vector<std::vector<double>> y_data(ndata, std::vector<double>(n_output,0.0));
    for(int i =0; i< ndata; i++){
        gen_xy(x_data[i], y_data[i], w_real);
    }

    // 训练循环
    for (int epoch = 0; epoch < 50; ++epoch) {
        double total_loss = 0.0;
        for(int i=0; i < ndata; i++){
            x->set_input(x_data[i]);
            y->set_input(y_data[i]);
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            w->update(learning_rate);
            total_loss += loss->item();
        }
        std::cout<< "Epoch " << epoch+1 << ", Loss: " << total_loss / ndata << std::endl;
    }
    
    std::cout << "\n训练完成!" << std::endl;
    std::cout << "最终参数: w:";
    w->print(std::cout,true);
    std::cout << "真实参数: w_real=[";
    for (size_t i = 0; i < w_real.size(); ++i) {
        std::cout << w_real[i];
        if (i < w_real.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 测试预测
    std::cout << "\n6. 预测测试:" << std::endl;
    auto test_x = std::vector<double>(n_input, 0.0);
    auto test_y = std::vector<double>(n_output, 0.0);
    gen_xy(test_x, test_y, w_real);
    x->set_input(test_x);
    yhat->calc();
    // y->set_data(test_y);
    // loss->forward();
    std::cout<< "yhat: ";
    yhat->print(std::cout,true);
    std::cout<< "y_real: ";
    for (const auto& val : test_y) {
        std::cout << val << " ";
    }
    return 0;
}