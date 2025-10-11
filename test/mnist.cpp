#include "../autograd.hpp"
#include <iostream>
#include <vector>
int main(){
    int n = 29;
    int n_input = n*n;
    int n_output = 10;
    int n_kernel = 32;
    int n_kernel_2 = 64;
    auto x = make_input(std::vector<double>(n_input, 0.0), {n, n});
    auto y = make_input(std::vector<double>(n_output, 0.0), {n_output});
    auto kernel = make_param(std::vector<double>(3*3 * n_kernel, 0.0), {3, 3,n_kernel});

    std::vector<VarPtr> nodes;
    for(int i = 0; i < n_kernel; i++){
        auto node = MaxPooling(relu(conv2d(x, slice(kernel, {-1, -1, i}))), 2);
        nodes.push_back(node);
    }
    auto new_layer = stack(nodes); // shape: (n_kernel, 14, 14 )

    auto kernel2 = make_param(std::vector<double>(3*3 * n_kernel * n_kernel_2, 0.0), {3, 3, n_kernel, n_kernel_2});
    nodes.clear();
    for(int i = 0; i < n_kernel_2; i++){
        std::vector<VarPtr> channels;
        for(int j = 0; j < n_kernel; j++){
            auto channel = conv2d(slice(new_layer, {j, -1, -1}), slice(kernel2, {-1, -1, j, i}));
            channels.push_back(channel);
        }
        auto node = channels[0];
        for(int k = 1; k < channels.size(); k++){
            node = node + channels[k];
        }

        auto node = MaxPooling(relu(node), 2);
        nodes.push_back(node);
    }
    auto new_layer_2 = stack(nodes); // shape: (n_kernel_2, 6, 6 )
    VarPtr flat;
    
    auto flat = reshape(new_layer_2, {n_kernel_2 * 6 * 6});
}