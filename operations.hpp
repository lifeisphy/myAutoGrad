/**
 * 基础数学运算函数
 */

// 加法运算
// get broadcasted index
#include <assert.h>
Edge* add_link(VarPtr parent, VarPtr child, bool updated=false){
    auto e = new Edge(parent, child, updated);
    parent->add_child(e);
    child->add_parent(e);
    return e;
}
std::vector<int> get_broadcast_idx(const std::vector<int>& result_idx, const std::vector<size_t>& var_shape) {
    int result_dims = result_idx.size();
    if (var_shape.empty()) return {};

    std::vector<int> var_idx(var_shape.size());
    int offset = result_dims - var_shape.size();
    
    for (size_t j = 0; j < var_shape.size(); j++) {
        int idx_val = (var_shape[j] == 1) ? 0 : result_idx[j + offset];
        var_idx[j] = idx_val;
    }
    return var_idx;
}


VarPtr add(VarPtr a, VarPtr b)
{

    size_t len = std::max(a->shape().size(), b->shape().size());
    std::vector<size_t> result_shape(len);
    for (size_t i=0; i < len; i++)
    { // 确定结果shape

        size_t a_dim = (i <  a->shape().size()) ? a->shape()[a->shape().size()-i-1] : 1;
        size_t b_dim = (i < b->shape().size()) ? b->shape()[b->shape().size()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::runtime_error("Incompatible sizes for addition");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    size_t result_size = 1; // 计算结果总大小
    for (size_t dim : result_shape)
    {
        result_size *= dim;
    }

    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    add_link(result, a);
    add_link(result, b);
    // 执行加法（支持广播）
    auto forward_fn = [result, result_size]() {
        for (size_t i = 0; i < result_size; i++)
        {
            double val = 0.0;
            std::vector<int> result_idx = result->PlainItemIndex(i);
            for(auto edge: result->children()){
                auto node = edge->child;
                if(node == nullptr) continue;
                std::vector<int> node_idx = get_broadcast_idx(result_idx, node->shape());
                double node_val= (node->shape().empty()) ? node->data()[0] : node->Item(node_idx);
                val += node_val;
            }
            result->Item(i) = val;
            // 直接使用 result 变量计算索引，避免创建临时变量
            
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [ result, result_size](const std::vector<double> &grad_output)
    {
        for(auto edge: result->children()){
            auto node = edge->child;
            if(node == nullptr) continue;
            if(node->has_grad()){
                std::vector<double> grad_node(node->size(), 0.0);
                for (size_t i = 0; i < result_size; i++)
                {
                    std::vector<int> node_idx = get_broadcast_idx(result->PlainItemIndex(i), node->shape());
                    grad_node[node->ItemIndex(node_idx)] += grad_output[i];
                }
                edge->updated = true;
                node->accumulate_gradient(grad_node);
            }
        }
    };
    result->set_grad_fn(grad_fn);

    return result;
}

// 减法运算
VarPtr sub(VarPtr a, VarPtr b){

    size_t len = std::max(a->shape().size(), b->shape().size());
    std::vector<size_t> result_shape(len);
    for (size_t i=0; i < len; i++)
    { // 确定结果shape

        size_t a_dim = (i <  a->shape().size()) ? a->shape()[a->shape().size()-i-1] : 1;
        size_t b_dim = (i < b->shape().size()) ? b->shape()[b->shape().size()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::runtime_error("Incompatible sizes for subtraction");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    size_t result_size = 1; // 计算结果总大小
    for (size_t dim : result_shape)
    {
        result_size *= dim;
    }

    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [a, b, result, result_size]() {
        for (size_t i = 0; i < result_size; i++)
        {
            // 直接使用 result 变量计算索引，避免创建临时变量
            std::vector<int> result_idx = result->PlainItemIndex(i);
            
            // 计算广播后的 a 和 b 索引
            std::vector<int> a_idx = get_broadcast_idx(result_idx, a->shape());
            std::vector<int> b_idx = get_broadcast_idx(result_idx, b->shape());
            
            double a_val = (a->shape().empty()) ? a->data()[0] : a->Item(a_idx);
            double b_val = (b->shape().empty()) ? b->data()[0] : b->Item(b_idx);
            result->Item(i) = a_val - b_val;
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto link_a = add_link(result, a);
    auto link_b = add_link(result, b);
    auto grad_fn = [link_a, link_b,a,b, result, result_size](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size(), 0.0);
            for (size_t i = 0; i < result_size; i++)
            {
                std::vector<int> a_idx = get_broadcast_idx(result->PlainItemIndex(i), a->shape());
                grad_a[a->ItemIndex(a_idx)] += grad_output[i];
            }
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
        if (b->has_grad())
        {
            std::vector<double> grad_b(b->size(), 0.0);
            for (size_t i = 0; i < result_size; i++)
            {
                std::vector<int> b_idx = get_broadcast_idx(result->PlainItemIndex(i), b->shape());
                grad_b[b->ItemIndex(b_idx)] -= grad_output[i];
            }
            link_b->updated = true;
            b->accumulate_gradient(grad_b);
        }
        for(auto edge: result->children()){
            edge->updated = true;
        }
    };

    result->set_grad_fn(grad_fn);

    

    return result;
}
// 乘法运算 - 支持张量乘法


// 元素级乘法
VarPtr mul_elementwise(VarPtr a, VarPtr b) {
    size_t len = std::max(a->shape().size(), b->shape().size());
    std::vector<size_t> result_shape(len);
    
    // 计算结果形状（从右对齐的广播）
    for (size_t i = 0; i < len; i++) {
        size_t a_dim = (i < a->shape().size()) ? a->shape()[a->shape().size()-i-1] : 1;
        size_t b_dim = (i < b->shape().size()) ? b->shape()[b->shape().size()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("Incompatible sizes for element-wise multiplication");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    
    size_t result_size = 1;
    for (size_t dim : result_shape) {
        result_size *= dim;
    }
    
    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [result, result_size]() {
        for (size_t i = 0; i < result_size; i++) {
            std::vector<int> result_idx = result->PlainItemIndex(i);
            double val = 1.0;
            for(auto edge: result->children()){
                auto node = edge->child;
                if(node == nullptr) continue;
                std::vector<int> node_idx = get_broadcast_idx(result_idx, node->shape());
                double node_val= (node->shape().empty()) ? node->data()[0] : node->Item(node_idx);
                val *= node_val;
            }
            result->Item(i) = val;
            // 直接使用 result 变量计算索引，避免创建临时变量
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [result, result_size](const std::vector<double> &grad_output) {
        auto get_val = [result](Edge* edge, const int i){
            auto node = edge->child;
            if(node == nullptr) return 1.0;
            std::vector<int> node_idx = get_broadcast_idx(result->PlainItemIndex(i), node->shape());
            double factor= (node->shape().empty()) ? node->data()[0] : node->Item(node_idx);
            return factor;
        };
        for(auto edge: result->children()){
            auto node = edge->child;
            if(node == nullptr || !node->has_grad()) continue;
            std::vector<double> grad_node(node->size(), 0.0);
            for (size_t i = 0; i < result_size; i++) {
                double node_val = grad_output[i];
                std::vector<int> node_idx = get_broadcast_idx(result->PlainItemIndex(i), node->shape());

                for(auto other_edge: result->children()){
                    if(other_edge == edge) continue;
                    node_val *= get_val(other_edge, i);
                }
                grad_node[node->ItemIndex(node_idx)] += node_val;
            }
            edge->updated = true;
            node->accumulate_gradient(grad_node);            
        }
    };
    result->set_grad_fn(grad_fn);
    add_link(result, a);
    add_link(result, b);

    
    return result;

}

VarPtr mul(VarPtr a, VarPtr b, int axis_a = -1, int axis_b = -1)
{
    if (axis_a == -1 && axis_b == -1) {
        return mul_elementwise(a, b);
    }
    // 张量乘法：沿指定轴收缩
    // 检查轴的有效性
    if (axis_a < 0) axis_a += a->shape().size();
    if (axis_b < 0) axis_b += b->shape().size();
    
    if (axis_a >= static_cast<int>(a->shape().size()) || axis_b >= static_cast<int>(b->shape().size()) ||
        axis_a < 0 || axis_b < 0) {
        throw std::runtime_error("Invalid axis for tensor multiplication");
    }
    
    size_t contract_dim_a = a->shape()[axis_a];
    size_t contract_dim_b = b->shape()[axis_b];
    
    if (contract_dim_a != contract_dim_b) {
        throw std::runtime_error("Incompatible dimensions for tensor contraction");
    }
    
    // 计算结果形状
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < a->shape().size(); i++) {
        if (i != static_cast<size_t>(axis_a)) {
            result_shape.push_back(a->shape()[i]);
        }
    }
    for (size_t i = 0; i < b->shape().size(); i++) {
        if (i != static_cast<size_t>(axis_b)) {
            result_shape.push_back(b->shape()[i]);
        }
    }
    
    if (result_shape.empty()) {
        result_shape = {1}; // 标量结果
    }
    
    size_t result_size = 1;
    for (size_t dim : result_shape) {
        result_size *= dim;
    }
    
    std::vector<double> result_data(result_size, 0.0);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [a, b, result, result_size, axis_a, axis_b, contract_dim_a]() {
        // 执行张量乘法
        for (size_t res_i = 0; res_i < result_size; res_i++) {
            std::vector<int> result_idx = result->PlainItemIndex(res_i);
            
            // 构造 a 和 b 的完整索引
            std::vector<int> full_a_idx(a->shape().size());
            std::vector<int> full_b_idx(b->shape().size());
            
            // 填充 a 的索引（跳过收缩轴）
            size_t a_idx_pos = 0;
            for (size_t i = 0; i < a->shape().size(); i++) {
                if (i != static_cast<size_t>(axis_a)) {
                    if (a_idx_pos < result_idx.size()) {
                        full_a_idx[i] = result_idx[a_idx_pos++];
                    }
                }
            }
            
            // 填充 b 的索引（跳过收缩轴）
            size_t b_idx_pos = a->shape().size() - 1; // b 的索引从 a 的维度之后开始
            if (static_cast<size_t>(axis_a) < a->shape().size()) {
                b_idx_pos--; // 因为跳过了 a 的收缩轴
            }
            
            for (size_t i = 0; i < b->shape().size(); i++) {
                if (i != static_cast<size_t>(axis_b)) {
                    if (b_idx_pos < result_idx.size()) {
                        full_b_idx[i] = result_idx[b_idx_pos++];
                    }
                }
            }
            
            // 初始化结果为0
            result->Item(res_i) = 0.0;
            // 沿收缩维度求和
            for (size_t k = 0; k < contract_dim_a; k++) {
                full_a_idx[axis_a] = k;
                full_b_idx[axis_b] = k;
                
                double a_val = a->Item(full_a_idx);
                double b_val = b->Item(full_b_idx);
                result->Item(res_i) += a_val * b_val;
            }
        }
    };
    
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    
    auto link_a = add_link(result, a);
    auto link_b = add_link(result, b);
    auto grad_fn = [link_a, link_b, a, b, result, axis_a, axis_b, contract_dim_a](const std::vector<double> &grad_output) {
        if (a->has_grad()) {
            std::vector<double> grad_a(a->size(), 0.0);
            // 实现张量乘法的反向传播
            // grad_a[i,j,k] = sum_l (grad_output[i,j,l] * b[k,l])  (假设沿最后一维收缩)
            // 这里需要根据具体的轴来计算
            for (size_t a_i = 0; a_i < a->size(); a_i++) {
                std::vector<int> a_idx = a->PlainItemIndex(a_i);
                
                for (size_t res_i = 0; res_i < result->size(); res_i++) {
                    std::vector<int> result_idx = result->PlainItemIndex(res_i);
                    
                    // 构造对应的 b 索引
                    std::vector<int> b_idx(b->shape().size());
                    b_idx[axis_b] = a_idx[axis_a];
                    
                    // 填充 b 的其他维度索引
                    size_t b_dim_pos = a->shape().size() - 1;
                    if (static_cast<size_t>(axis_a) < a->shape().size()) b_dim_pos--;
                    
                    for (size_t i = 0; i < b->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_b)) {
                            if (b_dim_pos < result_idx.size()) {
                                b_idx[i] = result_idx[b_dim_pos++];
                            }
                        }
                    }
                    
                    // 检查 a 的索引是否匹配结果索引
                    bool match = true;
                    size_t a_dim_pos = 0;
                    for (size_t i = 0; i < a->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_a)) {
                            if (a_dim_pos < result_idx.size() && a_idx[i] != result_idx[a_dim_pos]) {
                                match = false;
                                break;
                            }
                            a_dim_pos++;
                        }
                    }
                    
                    if (match) {
                        double b_val = b->Item(b_idx);
                        grad_a[a_i] += grad_output[res_i] * b_val;
                    }
                }
            }
            link_a ->updated = true;
            a->accumulate_gradient(grad_a);
        }
        
        if (b->has_grad()) {
            std::vector<double> grad_b(b->size(), 0.0);
            // 类似地计算 b 的梯度
            for (size_t b_i = 0; b_i < b->size(); b_i++) {
                std::vector<int> b_idx = b->PlainItemIndex(b_i);
                
                for (size_t res_i = 0; res_i < result->size(); res_i++) {
                    std::vector<int> result_idx = result->PlainItemIndex(res_i);
                    
                    // 构造对应的 a 索引
                    std::vector<int> a_idx(a->shape().size());
                    a_idx[axis_a] = b_idx[axis_b];
                    
                    // 填充 a 的其他维度索引
                    size_t a_dim_pos = 0;
                    for (size_t i = 0; i < a->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_a)) {
                            if (a_dim_pos < result_idx.size()) {
                                a_idx[i] = result_idx[a_dim_pos++];
                            }
                        }
                    }
                    
                    // 检查 b 的索引是否匹配结果索引
                    bool match = true;
                    size_t b_dim_pos = a->shape().size() - 1;
                    if (static_cast<size_t>(axis_a) < a->shape().size()) b_dim_pos--;
                    
                    for (size_t i = 0; i < b->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_b)) {
                            if (b_dim_pos < result_idx.size() && b_idx[i] != result_idx[b_dim_pos]) {
                                match = false;
                                break;
                            }
                            b_dim_pos++;
                        }
                    }
                    
                    if (match) {
                        double a_val = a->Item(a_idx);
                        grad_b[b_i] += grad_output[res_i] * a_val;
                    }
                }
            }
            link_b->updated = true;
            b->accumulate_gradient(grad_b);
        }
    };
        
    result->set_grad_fn(grad_fn);
    
    return result;
}

// VarPtr pow(VarPtr a, VarPtr b){
    //     if(a->is_matrix() ) {
        //         if(a->shape()[0] != a->shape()[1])
        //             throw std::runtime_error("Pow operation only supports square matrix inputs.");
        //         return mul(a,a,1,0); // 矩阵乘法
        //     }else if(a->is_scalar()){
            //     }
            // }
VarPtr pow_elementwise(VarPtr a, double exponent){
    
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data);
                
    // 添加前向函数
    auto forward_fn = [a, result, exponent]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::pow(a->data()[i], exponent);
        }
    };
    
    auto link_a = add_link(result, a);
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto grad_fn = [link_a, a, exponent](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // d/dx (x^n) = n * x^(n-1)
                grad_a[i] = grad_output[i] * exponent * std::pow(a->data()[i], exponent - 1);
            }
            link_a ->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };
    result->set_grad_fn(grad_fn);
    return result;
}
// 求和运算
VarPtr sum(VarPtr a)
{
    auto result = make_var(0.0);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        double sum_val = 0.0;
        for (double val : a->data())
        {
            sum_val += val;
        }
        result->Item(0) = sum_val;
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            // 梯度广播到所有元素
            std::vector<double> grad_a(a->size(), grad_output[0]);
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    

    return result;
}
VarPtr sum(std::vector<VarPtr> vars) {
    // 对多个变量求和
    // 所有变量的形状相同
    if (vars.empty()) {
        throw std::runtime_error("No variables to sum");
    }
    size_t expected_size = vars[0]->size();
    std::vector<size_t> expected_shape = vars[0]->shape();
    for (const auto& v : vars) {
        if (v->size() != expected_size || v->shape() != expected_shape) {
            throw std::runtime_error("All variables must have the same shape for summation");
        }
    }
    auto result = make_var(std::vector<double>(expected_size, 0.0), expected_shape);
    // 添加前向函数
    auto forward_fn = [vars, result]() {
        for(int i=0;i<result->size();i++){
            result->Item(i)=0.0;
            for(const auto& v: vars){
                result->Item(i)+=v->Item(i);
            }
        }
    };
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    for (auto& v : vars) {
        add_link(result, v);
    }
    auto grad_fn =[result](const std::vector<double> &grad_output) {
        for (const auto& e : result->children()) {
            if (e->child->has_grad()) {
                e->updated = true;
                e->child->accumulate_gradient(grad_output);
            }
        }
    };
    result->set_grad_fn(grad_fn);
    return result;
}
// 平均值运算
VarPtr mean(VarPtr a)
{
    auto result = make_var(0.0);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        double sum_val = 0.0;
        for (double val : a->data())
        {
            sum_val += val;
        }
        double mean_val = sum_val / a->size();
        result->Item(0) = mean_val;
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            // 梯度平均分配到所有元素
            double grad_per_element = grad_output[0] / a->size();
            std::vector<double> grad_a(a->size(), grad_per_element);
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    

    return result;
}

/**
 * 激活函数
 */

// ReLU激活函数
VarPtr relu(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data,a->shape());

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::max(0.0, a->data()[i]);
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // ReLU的导数：x > 0 时为1，否则为0
                grad_a[i] = (a->data()[i] > 0) ? grad_output[i] : 0.0;
            }
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);

    return result;
}

// Sigmoid激活函数
VarPtr sigmoid(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data,a->shape());

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = 1.0 / (1.0 + std::exp(-a->data()[i]));
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a, result](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // Sigmoid的导数：sigmoid(x) * (1 - sigmoid(x))
                double sigmoid_val = result->data()[i];
                grad_a[i] = grad_output[i] * sigmoid_val * (1.0 - sigmoid_val);
            }
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    return result;
}

// Tanh激活函数
VarPtr tanh_activation(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data,a->shape());

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::tanh(a->data()[i]);
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a, result](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // Tanh的导数：1 - tanh²(x)
                double tanh_val = result->data()[i];
                grad_a[i] = grad_output[i] * (1.0 - tanh_val * tanh_val);
            }
            link_a->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    return result;
}

/**
 * 损失函数
 */

// 均方误差损失函数
VarPtr mse_loss(VarPtr predictions, VarPtr targets)
{
    if (predictions->size() != targets->size())
    {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    // 计算 MSE = mean((predictions - targets)^2)
    auto diff = sub(predictions, targets);
    auto squared_diff = mul(diff, diff);
    return mean(squared_diff);
}

// 二元交叉熵损失函数
VarPtr binary_cross_entropy_loss(VarPtr predictions, VarPtr targets)
{
    if (predictions->size() != targets->size())
    {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    std::vector<double> loss_data(predictions->size());
    auto result = make_var(loss_data);

    // 添加前向函数
    auto forward_fn = [predictions, targets, result]() {
        for (size_t i = 0; i < predictions->size(); ++i)
        {
            double pred = predictions->data()[i];
            double target = targets->data()[i];
            // 防止log(0)
            pred = std::max(1e-15, std::min(1.0 - 1e-15, pred));
            // BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]
            result->Item(i) = -(target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_pred = add_link(result, predictions);
    auto link_tgt = add_link(result, targets);
    auto grad_fn = [link_pred, link_tgt, predictions, targets](const std::vector<double> &grad_output)
    {
        if (predictions->has_grad())
        {
            std::vector<double> grad_pred(predictions->size());
            for (size_t i = 0; i < predictions->size(); ++i)
            {
                double pred = predictions->data()[i];
                double target = targets->data()[i];

                // 防止除零
                pred = std::max(1e-15, std::min(1.0 - 1e-15, pred));

                // BCE对prediction的导数：-(target/pred - (1-target)/(1-pred))
                grad_pred[i] = grad_output[i] * (-(target / pred) + (1.0 - target) / (1.0 - pred));
            }
            link_pred->updated = true;
            predictions->accumulate_gradient(grad_pred);
        }
    };
    result->set_grad_fn(grad_fn);

    return mean(result); // 返回平均损失
}

/**
 * 切片函数 - 从多维张量中提取子张量
 * @param input 输入张量
 * @param indices 索引列表，-1表示提取该维度的所有内容
 * 例如：slice(tensor, {0, -1, 2}) 表示取第0行，所有列，第2个深度
 */
VarPtr slice(VarPtr input, const std::vector<int>& indices) {
    const auto& input_shape = input->shape();
    
    // 检查索引维度是否匹配
    if (indices.size() != input_shape.size()) {
        throw std::runtime_error("Number of indices must match tensor dimensions");
    }
    
    // 计算输出形状和有效索引
    std::vector<size_t> output_shape;
    std::vector<bool> is_slice_dim(indices.size(), false);  // 标记哪些维度被切片
    std::vector<int> fixed_indices(indices.size());         // 固定的索引值
    
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] == -1) {
            // -1 表示提取整个维度
            output_shape.push_back(input_shape[i]);
            is_slice_dim[i] = true;
        } else {
            // 检查索引有效性
            if (indices[i] < 0 || indices[i] >= static_cast<int>(input_shape[i])) {
                throw std::runtime_error("Index out of bounds for dimension " + std::to_string(i));
            }
            fixed_indices[i] = indices[i];
            is_slice_dim[i] = false;
        }
    }
    
    // 如果输出形状为空（所有维度都被固定），则输出是标量
    if (output_shape.empty()) {
        output_shape = {1};
    }
    
    // 计算输出大小
    size_t output_size = 1;
    for (size_t dim : output_shape) {
        output_size *= dim;
    }
    
    // 创建输出张量
    std::vector<double> output_data(output_size);
    auto result = make_var(output_data, output_shape);
    
    // 前向传播函数
    auto forward_fn = [input, result, indices, input_shape, output_shape, 
                       is_slice_dim, fixed_indices, output_size]() {
        
        for (size_t out_idx = 0; out_idx < output_size; out_idx++) {
            // 将输出的平坦索引转换为多维索引
            std::vector<int> output_multi_idx = result->PlainItemIndex(out_idx);
            
            // 构造输入张量的索引
            std::vector<int> input_multi_idx(input_shape.size());
            size_t output_dim_counter = 0;
            
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (is_slice_dim[i]) {
                    // 这个维度被切片，使用输出索引
                    if (output_multi_idx.size() == 1 && output_shape.size() > 1) {
                        // 处理标量输出的特殊情况
                        input_multi_idx[i] = 0;
                    } else if (output_dim_counter < output_multi_idx.size()) {
                        input_multi_idx[i] = output_multi_idx[output_dim_counter];
                        output_dim_counter++;
                    }
                } else {
                    // 这个维度被固定，使用固定索引
                    input_multi_idx[i] = fixed_indices[i];
                }
            }
            
            // 从输入张量获取值并设置到输出
            double value = input->Item(input_multi_idx);
            result->Item(out_idx) = value;
        }
    };
    
    result->set_forward_fn(forward_fn);
    auto link = add_link(result, input);   
    // 反向传播函数
    auto grad_fn = [link, input, result, indices, input_shape, output_shape,
                    is_slice_dim, fixed_indices, output_size]
                   (const std::vector<double>& grad_output) {
        
        if (input->has_grad()) {
            std::vector<double> input_grad(input->size(), 0.0);
            
            for (size_t out_idx = 0; out_idx < output_size; out_idx++) {
                // 将输出的平坦索引转换为多维索引
                std::vector<int> output_multi_idx = result->PlainItemIndex(out_idx);
                
                // 构造输入张量的索引（与前向传播相同的逻辑）
                std::vector<int> input_multi_idx(input_shape.size());
                size_t output_dim_counter = 0;
                
                for (size_t i = 0; i < input_shape.size(); i++) {
                    if (is_slice_dim[i]) {
                        if (output_multi_idx.size() == 1 && output_shape.size() > 1) {
                            input_multi_idx[i] = 0;
                        } else if (output_dim_counter < output_multi_idx.size()) {
                            input_multi_idx[i] = output_multi_idx[output_dim_counter];
                            output_dim_counter++;
                        }
                    } else {
                        input_multi_idx[i] = fixed_indices[i];
                    }
                }
                
                // 将梯度累加到对应的输入位置
                size_t input_flat_idx = input->ItemIndex(input_multi_idx);
                input_grad[input_flat_idx] += grad_output[out_idx];
            }
            link->updated = true;    
            input->accumulate_gradient(input_grad);
        }
    };
    
    result->set_grad_fn(grad_fn);
    return result;
}

// 2D卷积运算
VarPtr conv2d(VarPtr a, VarPtr b){
    if(!a->is_matrix() || !b->is_matrix())
        throw std::runtime_error("Conv2d operation only supports 2D tensor inputs.");
    if(a->shape()[0] < b->shape()[0] || a->shape()[1] < b->shape()[1])
        throw std::runtime_error("Kernel size must be smaller than input size.");
    size_t out_rows = a->shape()[0] - b->shape()[0] + 1;
    size_t out_cols = a->shape()[1] - b->shape()[1] + 1;
    std::vector<size_t> result_shape = {out_rows, out_cols};
    std::vector<double> result_data(out_rows * out_cols, 0.0);
    auto result = make_var(result_data, result_shape);
    // 添加前向函数
    auto forward_fn = [a, b, result, out_rows, out_cols]() {
        for (size_t i = 0; i < out_rows; ++i)
        {
            for (size_t j = 0; j < out_cols; ++j)
            {
                double sum_val = 0.0;
                for (size_t m = 0; m < b->shape()[0]; ++m)
                {
                    for (size_t n = 0; n < b->shape()[1]; ++n)
                    {
                        sum_val += a->Item({static_cast<int>(i + m), static_cast<int>(j + n)}) * b->Item({static_cast<int>(m), static_cast<int>(n)});
                    }
                }
                result->Item({static_cast<int>(i), static_cast<int>(j)}) = sum_val;
            }
        }
    };
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_a = add_link(result, a);
    auto link_b = add_link(result, b);
    auto grad_fn = [link_a, link_b, a, b, result, out_rows, out_cols](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size(), 0.0);
            for (size_t i = 0; i < out_rows; ++i)
            {
                for (size_t j = 0; j < out_cols; ++j)
                {
                    for (size_t m = 0; m < b->shape()[0]; ++m)
                    {
                        for (size_t n = 0; n < b->shape()[1]; ++n)
                        {
                            grad_a[(i + m) * a->shape()[1] + (j + n)] += grad_output[i * out_cols + j] * b->Item({static_cast<int>(m), static_cast<int>(n)});
                        }
                    }
                }
            }
            link_a ->updated = true;
            a->accumulate_gradient(grad_a);
        }
        if (b->has_grad())
        {
            std::vector<double> grad_b(b->size(), 0.0);
            for (size_t i = 0; i < out_rows; ++i)
            {
                for (size_t j = 0; j < out_cols; ++j)
                {
                    for (size_t m = 0; m < b->shape()[0]; ++m)
                    {
                        for (size_t n = 0; n < b->shape()[1]; ++n)
                        {
                            grad_b[m * b->shape()[1] + n] += grad_output[i * out_cols + j] * a->Item({static_cast<int>(i + m), static_cast<int>(j + n)});
                        }
                    }
                }
            }
            link_b->updated = true;
            b->accumulate_gradient(grad_b);
        }
    };

    result->set_grad_fn(grad_fn);
   
    return result;
}
VarPtr MaxPooling(VarPtr a, size_t filter_size=2){
    size_t stride = filter_size; // 默认步长等于滤波器大小
    if(!a->is_matrix())
        throw std::runtime_error("MaxPooling operation only supports 2D tensor inputs.");
    if(a->shape()[0] < 2 || a->shape()[1] < 2)
        throw std::runtime_error("Input size must be at least 2x2 for MaxPooling.");
    assert(stride == filter_size);
    size_t out_rows = a->shape()[0] / stride;
    size_t out_cols = a->shape()[1] / stride;
    std::vector<size_t> result_shape = {out_rows, out_cols};
    std::vector<double> result_data(out_rows * out_cols, 0.0);
    auto result = make_var(result_data, result_shape);
    // 添加前向函数
    auto forward_fn = [a, result, out_rows, out_cols, stride, filter_size]() {
        for (size_t i = 0; i < out_rows; ++i)
        {
            for (size_t j = 0; j < out_cols; ++j)
            {
                double max_val = -std::numeric_limits<double>::infinity();
                for (size_t m = 0; m < filter_size; ++m)
                {
                    for (size_t n = 0; n < filter_size; ++n)
                    {
                        double val;
                        if(i * stride + m >= a->shape()[0] || j * stride + n >= a->shape()[1])
                            val = 0; // 边界外补0
                        else
                            val = a->Item({static_cast<int>(i * stride + m), static_cast<int>(j * stride + n)});
                        if (val > max_val)
                        {
                            max_val = val;
                        }
                    }
                }
                result->Item({static_cast<int>(i), static_cast<int>(j)}) = max_val;
            }
        }
    };
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto link_a = add_link(result, a);
    auto grad_fn = [link_a, a, result, out_rows, out_cols, filter_size, stride](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size(), 0.0);
            for (int i = 0; i < out_rows; ++i)
            {
                for (int j = 0; j < out_cols; ++j)
                {
                    double max_val = result->Item({i, j});
                    for (int m = 0; m < filter_size; ++m)
                    {
                        for (int n = 0; n < filter_size; ++n)
                        {
                            if (a->Item({i * (int)stride + m, j * (int)stride + n}) == max_val)
                            {
                                grad_a[a->ItemIndex({i * (int)stride + m, j * (int)stride + n})] += grad_output[result->ItemIndex({i,j})];
                            }
                        }
                    }
                }
            }
            link_a ->updated = true;
            a->accumulate_gradient(grad_a);
        }
    };
    result->set_grad_fn(grad_fn);
    return result;
}

VarPtr stack(std::vector<VarPtr> vars){
    // input parameters must have the same shape
    // output shape = { num_vars, shape}
    if(vars.empty())
        throw std::runtime_error("Input list for stack operation is empty.");
    size_t num_vars = vars.size();
    size_t var_size = vars[0]->size();
    for(const auto& v : vars){
        if(v->size() != var_size)
            throw std::runtime_error("All variables must have the same size for stack operation.");
    }
    std::vector<size_t> result_shape = {num_vars};
    for(int dim: vars[0]->shape()){
        result_shape.push_back(dim);
    }
    // result_shape.insert(result_shape.end(), vars[0]->shape().begin(), vars[0]->shape().end());
    size_t result_size = num_vars * var_size;
    std::vector<double> result_data(result_size, 0.0);
    auto result = make_var(result_data, result_shape);
    // 添加前向函数
    auto forward_fn = [vars, result, num_vars, var_size]() {
        for(size_t i = 0; i < num_vars; ++i){
            for(size_t j = 0; j < var_size; ++j){
                result->Item(i * var_size + j) = vars[i]->data()[j];
            }
        }
    };
    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto grad_fn = [vars, result, num_vars, var_size](const std::vector<double> &grad_output)
    {
        for(int i=0; i<num_vars; ++i){
            auto edge = result->children()[i];
            auto node = edge->child;
            if(node->has_grad()){
                std::vector<double> grad_var(var_size, 0.0);
                for(size_t j = 0; j < var_size; ++j){
                    grad_var[j] = grad_output[i * var_size + j];
                }
                edge->updated = true;
                vars[i]->accumulate_gradient(grad_var);
            }
        }
    };
    result->set_grad_fn(grad_fn);
    for(const auto& v : vars){
        add_link(result, v);
    }
    return result;
}


