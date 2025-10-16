#include "../autograd.hpp"
#include "../utils.hpp"
#include <iostream>
#include <iomanip>
using namespace std;

void test_attention_mechanism() {
    cout << "=== 测试注意力机制中的slice和stack应用 ===" << endl;
    
    // 模拟一个简单的注意力机制
    // Query: 1x3, Key: 2x3, Value: 2x3
    auto query = make_var({1.0, 2.0, 3.0}, {1, 3});
    auto key = make_var({1.0, 0.0, 1.0, 0.0, 1.0, 0.0}, {2, 3});
    auto value = make_var({0.5, 1.5, 2.5, 1.0, 2.0, 3.0}, {2, 3});
    
    cout << "Query (1x3): "; print_vec(cout, query->data()); cout << endl;
    cout << "Key (2x3):   "; print_vec(cout, key->data()); cout << endl;
    cout << "Value (2x3): "; print_vec(cout, value->data()); cout << endl;
    
    // 计算注意力权重：每个key与query的点积 (简化版本)
    std::vector<VarPtr> attention_scores;
    for(int i = 0; i < 2; i++) {
        auto key_i = slice(key, {i, -1});  // 提取第i个key向量
        // 简化的点积计算：sum(query * key_i)
        auto product = mul_elementwise(query, key_i);
        auto score = sum(product);
        attention_scores.push_back(score);
        
        cout << "Key " << i << ": "; print_vec(cout, key_i->data());
        cout << " Score: " << score->data()[0] << endl;
    }
    
    // 使用slice和stack重组value矩阵
    auto val0 = slice(value, {0, -1});
    auto val1 = slice(value, {1, -1});
    auto stacked_values = stack({val0, val1});
    
    cout << "\n重组后的values: "; print_vec(cout, stacked_values->data()); cout << endl;
    cout << "形状: "; print_vec(cout, stacked_values->shape()); cout << endl;
}

void test_batch_processing() {
    cout << "\n=== 测试批处理中的slice和stack应用 ===" << endl;
    
    // 模拟批处理：3个样本，每个样本是2x2的图像
    auto batch = make_var({
        1,2,3,4,    // 样本1
        5,6,7,8,    // 样本2  
        9,10,11,12  // 样本3
    }, {3, 2, 2});
    
    cout << "批处理数据 (3x2x2): "; print_vec(cout, batch->data()); cout << endl;
    
    // 从批处理中提取单个样本
    std::vector<VarPtr> samples;
    for(int i = 0; i < 3; i++) {
        auto sample = slice(batch, {i, -1, -1});
        samples.push_back(sample);
        cout << "样本 " << i << " (2x2): "; print_vec(cout, sample->data()); cout << endl;
    }
    
    // 对每个样本进行处理（比如加上偏置）
    std::vector<VarPtr> processed_samples;
    for(int i = 0; i < 3; i++) {
        auto bias = make_var({(double)(i+1), (double)(i+1), (double)(i+1), (double)(i+1)}, {2, 2});  // 匹配形状的偏置
        auto processed = add(samples[i], bias);
        processed_samples.push_back(processed);
        cout << "处理后样本 " << i << ": "; print_vec(cout, processed->data()); cout << endl;
    }
    
    // 重新组成批处理
    auto new_batch = stack(processed_samples);
    cout << "\n重新组成的批处理: "; print_vec(cout, new_batch->data()); cout << endl;
    cout << "形状: "; print_vec(cout, new_batch->shape()); cout << endl;
}

void test_matrix_operations() {
    cout << "\n=== 测试矩阵操作中的slice和stack应用 ===" << endl;
    
    // 创建一个4x4矩阵
    auto matrix = make_var({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16
    }, {4, 4});
    
    cout << "原始4x4矩阵: "; print_vec(cout, matrix->data()); cout << endl;
    
    // 提取对角线元素
    std::vector<VarPtr> diagonal_elements;
    for(int i = 0; i < 4; i++) {
        auto diag_elem = slice(matrix, {i, i});
        diagonal_elements.push_back(diag_elem);
    }
    auto diagonal = stack(diagonal_elements);
    cout << "对角线元素: "; print_vec(cout, diagonal->data()); cout << endl;
    
    // 提取上三角部分（不包括对角线）
    std::vector<VarPtr> upper_triangle;
    for(int i = 0; i < 4; i++) {
        for(int j = i+1; j < 4; j++) {
            auto elem = slice(matrix, {i, j});
            upper_triangle.push_back(elem);
        }
    }
    auto upper_tri = stack(upper_triangle);
    cout << "上三角元素: "; print_vec(cout, upper_tri->data()); cout << endl;
    
    // 提取每行的最后一个元素
    std::vector<VarPtr> last_column;
    for(int i = 0; i < 4; i++) {
        auto last_elem = slice(matrix, {i, 3});
        last_column.push_back(last_elem);
    }
    auto last_col = stack(last_column);
    cout << "最后一列: "; print_vec(cout, last_col->data()); cout << endl;
}

void test_gradient_complex() {
    cout << "\n=== 测试复杂梯度传播 ===" << endl;
    
    // 创建参数矩阵
    auto params = make_param({1,2,3,4,5,6}, {2,3});
    
    // 提取不同部分并进行不同操作
    auto first_row = slice(params, {0, -1});     // [1,2,3]
    auto second_row = slice(params, {1, -1});    // [4,5,6]
    
    // 对第一行求和，对第二行求平方和
    auto sum1 = sum(first_row);                  // 1+2+3 = 6
    auto sum2 = sum(mul_elementwise(second_row, second_row)); // 16+25+36 = 77
    
    // 总损失
    auto total_loss = add(sum1, sum2);           // 6 + 77 = 83
    
    cout << "第一行: "; print_vec(cout, first_row->data()); cout << endl;
    cout << "第二行: "; print_vec(cout, second_row->data()); cout << endl;
    cout << "第一行求和: " << sum1->data()[0] << endl;
    cout << "第二行平方和: " << sum2->data()[0] << endl;
    cout << "总损失: " << total_loss->data()[0] << endl;
    
    // 反向传播
    total_loss->backward();
    
    cout << "\n梯度分析:" << endl;
    cout << "参数梯度: "; print_vec(cout, params->grad()); cout << endl;
    cout << "预期梯度: [1, 1, 1, 8, 10, 12]" << endl;
    cout << "解释: 第一行梯度都是1(线性求和), 第二行梯度是2*原值(平方函数)" << endl;
    
    // 验证梯度
    std::vector<double> expected = {1, 1, 1, 8, 10, 12};
    bool correct = true;
    auto actual = params->grad();
    for(size_t i = 0; i < expected.size(); i++) {
        if(abs(actual[i] - expected[i]) > 1e-6) {
            correct = false;
            break;
        }
    }
    cout << "梯度正确性: " << (correct ? "✓ 通过" : "✗ 失败") << endl;
}

int main() {
    cout << "测试slice和stack函数的实际应用场景..." << endl << endl;
    
    try {
        test_attention_mechanism();
        test_batch_processing();
        test_matrix_operations();
        test_gradient_complex();
        
        cout << "\n=== 所有实际应用测试完成! ===" << endl;
        
    } catch (const std::exception& e) {
        cout << "测试过程中发生错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}