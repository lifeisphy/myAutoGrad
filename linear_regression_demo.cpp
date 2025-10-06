#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "autograd.hpp"

// 生成随机数据的辅助函数
std::vector<double> random_vector(size_t size, double min_val = 0.0, double max_val = 1.0) {
    std::vector<double> vec(size);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

// 生成随机正态分布噪声
std::vector<double> random_normal(size_t size, double mean = 0.0, double std = 1.0) {
    std::vector<double> vec(size);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, std);
    
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

int main() {
    std::cout << "=== 线性回归优化示例: 最小化 |y - wx - b|² ===" << std::endl;
    
    // 数据集参数
    const size_t n_samples = 100;    // 样本数量
    const size_t n_features = 3;     // 特征维度
    const size_t n_outputs = 2;      // 输出维度
    
    // 真实参数 w0 和 b0（用于生成数据）
    auto w0_data = random_vector(n_features * n_outputs, -2.0, 2.0);
    auto b0_data = random_vector(n_outputs, -1.0, 1.0);
    
    auto w0 = make_var(w0_data, false, {n_features, n_outputs});
    auto b0 = make_var(b0_data, false, {n_outputs});
    
    std::cout << "\\n=== 数据生成 ===" << std::endl;
    std::cout << "真实权重 w0 (" << n_features << "x" << n_outputs << "): ";
    w0->print();
    std::cout << "真实偏置 b0 (" << n_outputs << ",): ";
    b0->print();
    
    // 生成训练数据
    std::vector<VarPtr> X_list, Y_list;
    
    for (size_t i = 0; i < n_samples; ++i) {
        // 生成随机输入 x
        auto x_data = random_vector(n_features, -3.0, 3.0);
        auto x = make_var(x_data, false, {n_features});
        
        // 计算真实输出: y = w0 @ x + b0 + noise
        // auto wx = mul(w0, x, 0, 0);  // w0.T @ x (矩阵-向量乘法)
        // auto y_clean = add(wx, b0);   // 添加偏置
        auto y_clean = w0*x+b0;
        // 添加噪声
        auto noise_data = random_normal(n_outputs, 0.0, 0.1);
        auto noise = make_var(noise_data, false, {n_outputs});
        auto y = add(y_clean, noise);
        
        X_list.push_back(x);
        Y_list.push_back(y);
    }
    
    std::cout << "生成了 " << n_samples << " 个训练样本" << std::endl;
    
    // 初始化可训练参数
    auto w_data = random_vector(n_features * n_outputs, -0.5, 0.5);
    auto b_data = random_vector(n_outputs, -0.1, 0.1);
    
    auto w = make_var(w_data, true, {n_features, n_outputs});
    auto b = make_var(b_data, true, {n_outputs});
    
    std::cout << "\\n=== 初始参数 ===" << std::endl;
    std::cout << "初始权重 w: ";
    w->print();
    std::cout << "初始偏置 b: ";
    b->print();
    
    // 训练参数
    const double learning_rate = 0.01;
    const int max_epochs = 1000;
    const double tolerance = 1e-6;
    
    std::cout << "\\n=== 开始训练 ===" << std::endl;
    std::cout << "学习率: " << learning_rate << std::endl;
    std::cout << "最大迭代数: " << max_epochs << std::endl;
    
    double prev_loss = std::numeric_limits<double>::infinity();
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        // 清零梯度
        w->zero_grad();
        b->zero_grad();
        
        // 计算总损失
        VarPtr total_loss = make_var(0.0, true);
        
        for (size_t i = 0; i < n_samples; ++i) {
            // 前向传播: y_pred = w @ x + b
            auto wx = mul(w, X_list[i], 0, 0);  // w.T @ x
            auto y_pred = add(wx, b);
            
            // 计算损失: (y - y_pred)²
            auto diff = sub(Y_list[i], y_pred);
            auto squared_diff = mul(diff, diff);  // 元素级平方
            auto sample_loss = sum(squared_diff); // 对输出维度求和
            
            total_loss = add(total_loss, sample_loss);
        }
        
        // 平均损失
        auto avg_loss = mul(total_loss, make_var(1.0 / n_samples, false));
        
        // 反向传播
        avg_loss->backward();
        
        // 参数更新
        w->update(learning_rate);
        b->update(learning_rate);
        
        double current_loss = avg_loss->item();
        
        // 打印训练进度
        if (epoch % 100 == 0 || epoch == max_epochs - 1) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << ", Loss: " << std::fixed << std::setprecision(6) << current_loss << std::endl;
        }
        
        // 检查收敛
        if (std::abs(prev_loss - current_loss) < tolerance) {
            std::cout << "训练收敛于 epoch " << epoch << std::endl;
            break;
        }
        
        prev_loss = current_loss;
    }
    
    std::cout << "\\n=== 训练结果 ===" << std::endl;
    std::cout << "最终权重 w: ";
    w->print();
    std::cout << "最终偏置 b: ";
    b->print();
    
    std::cout << "\\n=== 与真实参数对比 ===" << std::endl;
    std::cout << "真实权重 w0: ";
    w0->print();
    std::cout << "训练权重 w:  ";
    w->print();
    
    std::cout << "真实偏置 b0: ";
    b0->print();
    std::cout << "训练偏置 b:  ";
    b->print();
    
    // 计算参数误差
    auto w_diff = sub(w, w0);
    auto b_diff = sub(b, b0);
    auto w_error = sum(mul(w_diff, w_diff));
    auto b_error = sum(mul(b_diff, b_diff));
    
    std::cout << "\\n权重误差 (MSE): " << w_error->item() << std::endl;
    std::cout << "偏置误差 (MSE): " << b_error->item() << std::endl;
    
    // 测试预测
    std::cout << "\\n=== 预测测试 ===" << std::endl;
    auto test_x = make_var({1.0, 2.0, 3.0}, false, {n_features});
    std::cout << "测试输入 x: ";
    test_x->print();
    
    // 真实输出
    auto true_wx = mul(w0, test_x, 0, 0);
    auto true_y = add(true_wx, b0);
    std::cout << "真实输出 y_true: ";
    true_y->print();
    
    // 预测输出
    auto pred_wx = mul(w, test_x, 0, 0);
    auto pred_y = add(pred_wx, b);
    std::cout << "预测输出 y_pred: ";
    pred_y->print();
    
    // 预测误差
    auto pred_diff = sub(pred_y, true_y);
    auto pred_error = sum(mul(pred_diff, pred_diff));
    std::cout << "预测误差 (MSE): " << pred_error->item() << std::endl;
    
    std::cout << "\\n=== 优化完成 ===" << std::endl;
    
    return 0;
}