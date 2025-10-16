#include "../autograd.hpp"
#include "../utils.hpp"
#include <iomanip>
using namespace std;

std::vector<double> vec(size_t size){
    std::vector<double> data(size);
    for(size_t i=0; i < size; i++){
        data[i] = i + 1;  // 避免全零输入
    }
    return data;
}

std::vector<double> vec_r(size_t size, double scale = 0.1){
    std::vector<double> data(size);
    static bool seeded = false;
    if (!seeded) {
        srand(42);  // 固定随机种子便于调试
        seeded = true;
    }
    for(size_t i=0; i < size; i++){
        data[i] = (rand() / double(RAND_MAX) - 0.5) * 2.0 * scale;  // [-scale, scale]
    }
    return data;
}
int main(){
    cout << "开始神经网络训练..." << endl;
    
    // 创建固定的输入和目标权重
    auto x = make_input(vec(4), {4});
    auto y = make_input(vec(3), {3});
    auto W = make_param(vec_r(4 * 3, 0.1), {4, 3});  // 小初始化
    
    // 真实权重矩阵
    auto w0 = vec_r(4 * 3, 0.5);
    cout << "真实权重矩阵 w0:" << endl;
    print_vec(cout, w0);
    
    // 目标函数：W0 * x 
    auto f = [w0](std::vector<double> x_val){
        std::vector<double> ret(3, 0.0);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 4; j++){
                ret[i] += w0[j*3+i] * x_val[j];  // W[j,i] = w0[j*3+i]
            }
        }
        return ret;
    };
    
    // 构建计算图（在循环外定义）
    auto yhat = mul(W, x, 0, 0);  // W * x
    auto loss = mse_loss(yhat, y);
    
    cout << "\n开始训练 (使用多个随机输入):" << endl;
    cout << "Epoch\tLoss\t输入样本" << endl;
    
    double lr = 0.001;  // 学习率
    int total_epochs = 1000;
    int samples_per_epoch = 5;  // 每个epoch使用5个随机样本
    
    for(int epoch = 0; epoch < total_epochs; epoch++){
        double epoch_loss = 0.0;
        
        // 每个epoch使用多个随机输入样本
        for(int sample = 0; sample < samples_per_epoch; sample++){
            // 生成随机输入
            vector<double> x_train = vec_r(4, 2.0);  // 范围 [-2, 2] 的随机输入
            vector<double> y_target = f(x_train);
            
            // 设置输入数据
            x->set_input(x_train);
            y->set_input(y_target);
            
            // 前向传播
            loss->zero_grad_recursive();
            loss->calc();
            
            epoch_loss += loss->data()[0];
            
            // 反向传播和参数更新
            loss->backward();
            W->update(lr);
        }
        
        // 计算平均损失
        epoch_loss /= samples_per_epoch;
        
        if(epoch % 100 == 0){
            cout << epoch << "\t" << fixed << setprecision(6) << epoch_loss;
            
            // 显示当前样本
            vector<double> current_x = vec_r(4, 2.0);
            x->set_input(current_x);
            y->set_input(f(current_x));
            loss->zero_grad_recursive();
            loss->calc();
            
            cout << "\t输入: [";
            for(int i = 0; i < current_x.size(); i++){
                if(i > 0) cout << ", ";
                cout << fixed << setprecision(2) << current_x[i];
            }
            cout << "]" << endl;
            
            if(epoch % 200 == 0){
                cout << "\t\t当前预测: ";
                print_vec(cout, yhat->data());
                cout << "\t目标输出: ";
                print_vec(cout, y->data());
                cout << endl;
            }
        }
    }
    
    cout << "\n训练完成!" << endl;
    cout << "真实权重矩阵:" << endl;
    print_vec(cout, w0);
    cout << "学习到的权重矩阵:" << endl; 
    print_vec(cout, W->data());
    
    // 最终测试：用多个测试样本验证
    cout << "\n=== 最终测试 (多个随机样本) ===" << endl;
    double total_test_loss = 0.0;
    int test_samples = 10;
    
    for(int test = 0; test < test_samples; test++){
        vector<double> test_x = vec_r(4, 2.0);  // 测试输入
        vector<double> expected_y = f(test_x);  // 期望输出
        
        x->set_input(test_x);
        y->set_input(expected_y);
        loss->zero_grad_recursive();
        loss->calc();
        
        total_test_loss += loss->data()[0];
        
        if(test < 3){  // 只显示前3个测试样本的详细结果
            cout << "测试 " << test+1 << ":" << endl;
            cout << "  输入: ";
            print_vec(cout, test_x);
            cout << "  预测: ";  
            print_vec(cout, yhat->data());
            cout << "  期望: ";
            print_vec(cout, expected_y);
            cout << "  误差: " << fixed << setprecision(6) << loss->data()[0] << endl;
        }
    }
    
    cout << "\n平均测试损失: " << fixed << setprecision(6) << total_test_loss / test_samples << endl;
    
    // 权重对比分析
    cout << "\n=== 权重对比分析 ===" << endl;
    cout << "真实权重 vs 学习权重 (按行显示):" << endl;
    for(int i = 0; i < 4; i++){
        cout << "第" << i+1 << "行: ";
        cout << "真实[";
        for(int j = 0; j < 3; j++){
            if(j > 0) cout << ", ";
            cout << fixed << setprecision(4) << w0[i*3+j];
        }
        cout << "] vs 学习[";
        for(int j = 0; j < 3; j++){
            if(j > 0) cout << ", ";
            cout << fixed << setprecision(4) << W->data()[i*3+j];
        }
        cout << "]" << endl;
    }
    
    return 0;
}