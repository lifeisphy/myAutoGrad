#include "../autograd.hpp"
#include "../utils.hpp"
#include <iostream>
#include <iomanip>
using namespace std;

// 简单的辅助函数来生成测试数据
std::vector<double> range_vec(size_t start, size_t size) {
    std::vector<double> data(size);
    for(size_t i = 0; i < size; i++) {
        data[i] = start + i;
    }
    return data;
}

void test_slice_basic() {
    cout << "=== 测试基本slice功能 ===" << endl;
    
    // 创建一个2x3的矩阵
    // [[1, 2, 3],
    //  [4, 5, 6]]
    auto matrix = make_var(range_vec(1, 6), {2, 3});
    cout << "原始矩阵 (2x3):" << endl;
    matrix->print(cout, true);
    
    // 测试1：取第0行（slice(matrix, {0, -1})）
    cout << "\n测试1：取第0行 slice({0, -1})" << endl;
    auto row0 = slice(matrix, {0, -1});
    cout << "预期结果: [1, 2, 3]" << endl;
    cout << "实际结果: ";
    print_vec(cout, row0->data());
    cout << endl;
    
    // 测试2：取第1列（slice(matrix, {-1, 1})）
    cout << "\n测试2：取第1列 slice({-1, 1})" << endl;
    auto col1 = slice(matrix, {-1, 1});
    cout << "预期结果: [2, 5]" << endl;
    cout << "实际结果: ";
    print_vec(cout, col1->data());
    cout << endl;
    
    // 测试3：取单个元素（slice(matrix, {1, 2})）
    cout << "\n测试3：取单个元素 slice({1, 2})" << endl;
    auto element = slice(matrix, {1, 2});
    cout << "预期结果: [6]" << endl;
    cout << "实际结果: ";
    print_vec(cout, element->data());
    cout << endl;
}

void test_slice_3d() {
    cout << "\n=== 测试3D张量slice功能 ===" << endl;
    
    // 创建一个2x2x3的3D张量
    // [[[1, 2, 3], [4, 5, 6]],
    //  [[7, 8, 9], [10, 11, 12]]]
    auto tensor3d = make_var(range_vec(1, 12), {2, 2, 3});
    cout << "原始3D张量 (2x2x3):" << endl;
    tensor3d->print(cout, true);
    
    // 测试：取第0个"切片"（slice(tensor3d, {0, -1, -1})）
    cout << "\n测试：取第0个2x3切片 slice({0, -1, -1})" << endl;
    auto slice_0 = slice(tensor3d, {0, -1, -1});
    cout << "预期结果: [[1, 2, 3], [4, 5, 6]]" << endl;
    cout << "实际结果: ";
    print_vec(cout, slice_0->data());
    cout << " 形状: ";
    print_vec(cout, slice_0->shape());
    cout << endl;
    
    // 测试：取特定元素（slice(tensor3d, {1, 0, 2})）
    cout << "\n测试：取特定元素 slice({1, 0, 2})" << endl;
    auto specific_element = slice(tensor3d, {1, 0, 2});
    cout << "预期结果: [9]" << endl;
    cout << "实际结果: ";
    print_vec(cout, specific_element->data());
    cout << endl;
}

void test_stack_basic() {
    cout << "\n=== 测试基本stack功能 ===" << endl;
    
    // 创建三个1x2向量
    auto vec1 = make_var({1.0, 2.0}, {2});
    auto vec2 = make_var({3.0, 4.0}, {2});
    auto vec3 = make_var({5.0, 6.0}, {2});
    
    cout << "输入向量:" << endl;
    cout << "vec1: "; print_vec(cout, vec1->data()); cout << endl;
    cout << "vec2: "; print_vec(cout, vec2->data()); cout << endl;
    cout << "vec3: "; print_vec(cout, vec3->data()); cout << endl;
    
    // 测试stack
    auto stacked = stack({vec1, vec2, vec3});
    cout << "\nstack后的结果:" << endl;
    cout << "预期结果: [[1, 2], [3, 4], [5, 6]]" << endl;
    cout << "实际结果: ";
    print_vec(cout, stacked->data());
    cout << " 形状: ";
    print_vec(cout, stacked->shape());
    cout << endl;
}

void test_stack_matrices() {
    cout << "\n=== 测试矩阵stack功能 ===" << endl;
    
    // 创建两个2x2矩阵
    auto mat1 = make_var({1.0, 2.0, 3.0, 4.0}, {2, 2});
    auto mat2 = make_var({5.0, 6.0, 7.0, 8.0}, {2, 2});
    
    cout << "输入矩阵:" << endl;
    cout << "mat1: "; print_vec(cout, mat1->data()); cout << endl;
    cout << "mat2: "; print_vec(cout, mat2->data()); cout << endl;
    
    // 测试stack
    auto stacked_mats = stack({mat1, mat2});
    cout << "\nstack后的结果:" << endl;
    cout << "预期形状: [2, 2, 2] (2个2x2矩阵)" << endl;
    cout << "实际形状: ";
    print_vec(cout, stacked_mats->shape());
    cout << "\n预期结果: [1, 2, 3, 4, 5, 6, 7, 8]" << endl;
    cout << "实际结果: ";
    print_vec(cout, stacked_mats->data());
    cout << endl;
}

void test_slice_stack_roundtrip() {
    cout << "\n=== 测试slice和stack的往返操作 ===" << endl;
    
    // 创建一个3x2矩阵
    auto original = make_var({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {3, 2});
    cout << "原始矩阵 (3x2): ";
    print_vec(cout, original->data());
    cout << endl;
    
    // 使用slice分离每一行
    auto row0 = slice(original, {0, -1});
    auto row1 = slice(original, {1, -1});
    auto row2 = slice(original, {2, -1});
    
    cout << "分离的行:" << endl;
    cout << "row0: "; print_vec(cout, row0->data()); cout << endl;
    cout << "row1: "; print_vec(cout, row1->data()); cout << endl;
    cout << "row2: "; print_vec(cout, row2->data()); cout << endl;
    
    // 使用stack重新组合
    auto reconstructed = stack({row0, row1, row2});
    cout << "\n重新stack后的结果:" << endl;
    cout << "形状: ";
    print_vec(cout, reconstructed->shape());
    cout << "\n数据: ";
    print_vec(cout, reconstructed->data());
    cout << endl;
    
    // 验证数据是否一致
    bool data_match = true;
    auto orig_data = original->data();
    auto recon_data = reconstructed->data();
    
    for(size_t i = 0; i < orig_data.size(); i++) {
        if(abs(orig_data[i] - recon_data[i]) > 1e-6) {
            data_match = false;
            break;
        }
    }
    
    cout << "数据一致性检查: " << (data_match ? "通过" : "失败") << endl;
}

void test_gradient_flow() {
    cout << "\n=== 测试梯度传播 ===" << endl;
    
    // 创建一个需要梯度的2x2矩阵
    auto matrix = make_param({1.0, 2.0, 3.0, 4.0}, {2, 2});
    
    // 使用slice提取第一行
    auto first_row = slice(matrix, {0, -1});
    
    // 计算第一行的和
    auto sum_result = sum(first_row);
    
    cout << "原始矩阵: ";
    print_vec(cout, matrix->data());
    cout << "\n第一行: ";
    print_vec(cout, first_row->data());
    cout << "\n第一行的和: " << sum_result->data()[0] << endl;
    
    // 反向传播
    sum_result->backward();
    
    cout << "\n反向传播后的梯度:" << endl;
    cout << "matrix梯度: ";
    print_vec(cout, matrix->grad());
    cout << "\n预期梯度: [1, 1, 0, 0] (只有第一行参与计算)" << endl;
    
    // 验证梯度
    auto expected_grad = std::vector<double>{1.0, 1.0, 0.0, 0.0};
    bool grad_correct = true;
    auto actual_grad = matrix->grad();
    
    for(size_t i = 0; i < expected_grad.size(); i++) {
        if(abs(actual_grad[i] - expected_grad[i]) > 1e-6) {
            grad_correct = false;
            break;
        }
    }
    
    cout << "梯度正确性检查: " << (grad_correct ? "通过" : "失败") << endl;
}

int main() {
    cout << "开始测试slice和stack函数..." << endl;
    
    try {
        test_slice_basic();
        test_slice_3d();
        test_stack_basic();
        test_stack_matrices();
        test_slice_stack_roundtrip();
        test_gradient_flow();
        
        cout << "\n=== 所有测试完成! ===" << endl;
        
    } catch (const std::exception& e) {
        cout << "测试过程中发生错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}