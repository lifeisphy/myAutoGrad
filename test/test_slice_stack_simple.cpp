#include "../autograd.hpp"
#include "../utils.hpp"
#include <iostream>
#include <cassert>
using namespace std;

// 简化的测试结果检查函数
bool check_data_equal(const DataView& actual, const std::vector<double>& expected, double tolerance = 1e-6) {
    if(actual.size() != expected.size()) return false;
    for(size_t i = 0; i < expected.size(); i++) {
        if(abs(actual[i] - expected[i]) > tolerance) return false;
    }
    return true;
}

bool check_shape_equal(const std::vector<size_t>& actual, const std::vector<size_t>& expected) {
    return actual == expected;
}

void run_slice_tests() {
    cout << "=== Slice Function Tests ===" << endl;
    int passed = 0, total = 0;
    
    // Test 1: 2D matrix row slice
    total++;
    auto matrix = make_var({1,2,3,4,5,6}, {2,3});
    auto row = slice(matrix, {0, -1});
    if(check_data_equal(row->data(), {1,2,3}) && check_shape_equal(row->shape(), {3})) {
        cout << "✓ Test 1: 2D row slice" << endl;
        passed++;
    } else {
        cout << "✗ Test 1: 2D row slice" << endl;
    }
    
    // Test 2: 2D matrix column slice  
    total++;
    auto col = slice(matrix, {-1, 1});
    if(check_data_equal(col->data(), {2,5}) && check_shape_equal(col->shape(), {2})) {
        cout << "✓ Test 2: 2D column slice" << endl;
        passed++;
    } else {
        cout << "✗ Test 2: 2D column slice" << endl;
    }
    
    // Test 3: Single element slice
    total++;
    auto element = slice(matrix, {1, 2});
    if(check_data_equal(element->data(), {6}) && check_shape_equal(element->shape(), {1})) {
        cout << "✓ Test 3: Single element slice" << endl;
        passed++;
    } else {
        cout << "✗ Test 3: Single element slice" << endl;
    }
    
    // Test 4: 3D tensor slice
    total++;
    auto tensor3d = make_var({1,2,3,4,5,6,7,8,9,10,11,12}, {2,2,3});
    auto slice3d = slice(tensor3d, {0, -1, -1});
    if(check_data_equal(slice3d->data(), {1,2,3,4,5,6}) && check_shape_equal(slice3d->shape(), {2,3})) {
        cout << "✓ Test 4: 3D tensor slice" << endl;
        passed++;
    } else {
        cout << "✗ Test 4: 3D tensor slice" << endl;
    }
    
    cout << "Slice tests: " << passed << "/" << total << " passed" << endl << endl;
}

void run_stack_tests() {
    cout << "=== Stack Function Tests ===" << endl;
    int passed = 0, total = 0;
    
    // Test 1: Stack vectors
    total++;
    auto v1 = make_var({1,2}, {2});
    auto v2 = make_var({3,4}, {2});
    auto v3 = make_var({5,6}, {2});
    auto stacked_vecs = stack({v1, v2, v3});
    if(check_data_equal(stacked_vecs->data(), {1,2,3,4,5,6}) && 
       check_shape_equal(stacked_vecs->shape(), {3,2})) {
        cout << "✓ Test 1: Stack vectors" << endl;
        passed++;
    } else {
        cout << "✗ Test 1: Stack vectors" << endl;
    }
    
    // Test 2: Stack matrices
    total++;
    auto m1 = make_var({1,2,3,4}, {2,2});
    auto m2 = make_var({5,6,7,8}, {2,2});
    auto stacked_mats = stack({m1, m2});
    if(check_data_equal(stacked_mats->data(), {1,2,3,4,5,6,7,8}) && 
       check_shape_equal(stacked_mats->shape(), {2,2,2})) {
        cout << "✓ Test 2: Stack matrices" << endl;
        passed++;
    } else {
        cout << "✗ Test 2: Stack matrices" << endl;
    }
    
    // Test 3: Stack single elements (scalars)
    total++;
    auto s1 = make_var({1.5}, {1});
    auto s2 = make_var({2.5}, {1});
    auto s3 = make_var({3.5}, {1});
    auto stacked_scalars = stack({s1, s2, s3});
    if(check_data_equal(stacked_scalars->data(), {1.5,2.5,3.5}) && 
       check_shape_equal(stacked_scalars->shape(), {3,1})) {
        cout << "✓ Test 3: Stack scalars" << endl;
        passed++;
    } else {
        cout << "✗ Test 3: Stack scalars" << endl;
    }
    
    cout << "Stack tests: " << passed << "/" << total << " passed" << endl << endl;
}

void run_integration_tests() {
    cout << "=== Integration Tests ===" << endl;
    int passed = 0, total = 0;
    
    // Test 1: Slice + Stack roundtrip
    total++;
    auto original = make_var({1,2,3,4,5,6}, {3,2});
    auto r0 = slice(original, {0,-1});
    auto r1 = slice(original, {1,-1});
    auto r2 = slice(original, {2,-1});
    auto reconstructed = stack({r0, r1, r2});
    if(check_data_equal(reconstructed->data(), {1,2,3,4,5,6}) && 
       check_shape_equal(reconstructed->shape(), {3,2})) {
        cout << "✓ Test 1: Slice + Stack roundtrip" << endl;
        passed++;
    } else {
        cout << "✗ Test 1: Slice + Stack roundtrip" << endl;
    }
    
    // Test 2: Gradient flow through slice
    total++;
    auto param = make_param({1,2,3,4}, {2,2});
    auto sliced = slice(param, {0,-1});
    auto loss = sum(sliced);
    loss->backward();
    if(check_data_equal(param->grad(), {1,1,0,0})) {
        cout << "✓ Test 2: Gradient flow through slice" << endl;
        passed++;
    } else {
        cout << "✗ Test 2: Gradient flow through slice" << endl;
    }
    
    // Test 3: Multiple slices from same tensor
    total++;
    auto tensor = make_var({1,2,3,4,5,6,7,8,9}, {3,3});
    auto diag1 = slice(tensor, {0,0});  // [1]
    auto diag2 = slice(tensor, {1,1});  // [5]  
    auto diag3 = slice(tensor, {2,2});  // [9]
    auto diagonal = stack({diag1, diag2, diag3});
    if(check_data_equal(diagonal->data(), {1,5,9}) && 
       check_shape_equal(diagonal->shape(), {3,1})) {
        cout << "✓ Test 3: Extract diagonal elements" << endl;
        passed++;
    } else {
        cout << "✗ Test 3: Extract diagonal elements" << endl;
    }
    
    cout << "Integration tests: " << passed << "/" << total << " passed" << endl << endl;
}

void run_error_tests() {
    cout << "=== Error Handling Tests ===" << endl;
    int passed = 0, total = 0;
    
    // Test 1: Index out of bounds
    total++;
    try {
        auto matrix = make_var({1,2,3,4}, {2,2});
        auto bad_slice = slice(matrix, {3, 0});  // 3 is out of bounds for dim 0
        cout << "✗ Test 1: Should throw for out of bounds index" << endl;
    } catch(const std::exception& e) {
        cout << "✓ Test 1: Correctly throws for out of bounds index" << endl;
        passed++;
    }
    
    // Test 2: Dimension mismatch in slice
    total++;
    try {
        auto matrix = make_var({1,2,3,4}, {2,2});
        auto bad_slice = slice(matrix, {0});  // Too few indices
        cout << "✗ Test 2: Should throw for dimension mismatch" << endl;
    } catch(const std::exception& e) {
        cout << "✓ Test 2: Correctly throws for dimension mismatch" << endl;
        passed++;
    }
    
    // Test 3: Empty stack
    total++;
    try {
        std::vector<VarPtr> empty_vars;
        auto bad_stack = stack(empty_vars);
        cout << "✗ Test 3: Should throw for empty stack" << endl;
    } catch(const std::exception& e) {
        cout << "✓ Test 3: Correctly throws for empty stack" << endl;
        passed++;  
    }
    
    cout << "Error handling tests: " << passed << "/" << total << " passed" << endl << endl;
}

void multiple_slice_backprop(){
    cout << "=== Multiple Slice Backpropagation Test ===" << endl;
    int passed = 0, total = 0;

    // Test: Multiple slices from same tensor with backprop
    total++;
    auto param = make_param({1,2,3,4,5,6}, {3,2});
    auto s0 = slice(param, {0,-1}); // [1,2]
    auto s1 = slice(param, {1,-1}); // [3,4]
    auto s2 = slice(param, {2,-1}); // [5,6]
    auto s3 = slice(s0, {0}); // [1]
    // auto sum_slices = add(add(s0, s1), s2); // sum of all slices
    auto loss = s3 + s3; // scalar loss
    loss->backward();
    if(check_data_equal(param->grad(), {2,0,0,0,0,0})) {
        cout << "✓ Test: Gradient flow through multiple slices" << endl;
        passed++;
    } else {
        cout << "✗ Test: Gradient flow through multiple slices" << endl;
    }
    
    total++;
    auto loss2 = sum(s1 + s3);
    loss2->zero_grad_recursive();
    loss2->backward();
    if(check_data_equal(param->grad(), {2,0,1,1,0,0})) {
        cout << "✓ Test: Gradient flow through different slices" << endl;
        passed++;
    } else {
        cout << "✗ Test: Gradient flow through different slices" << endl;
    }
    cout << "Multiple slice backprop tests: " << passed << "/" << total << " passed" << endl << endl;
}

int main() {
    cout << "Running comprehensive slice and stack function tests..." << endl << endl;
    
    run_slice_tests();
    run_stack_tests(); 
    run_integration_tests();
    run_error_tests();
    multiple_slice_backprop();
    cout << "=== Test Summary ===" << endl;
    cout << "All test suites completed. Check individual results above." << endl;
    
    return 0;
}