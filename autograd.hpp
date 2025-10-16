/**
 * 基础自动微分框架 (Basic Autograd Framework) - C++版本
 * 支持标量和向量的自动微分计算
 */
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <iomanip>
#include <ranges>
#include <unordered_set>
#include <fstream>
#include <map>
#include <cassert>
// 前向声明
class Variable;
class DataView;

using VarPtr = std::shared_ptr<Variable>;
using string = std::string;

/**
 * 支持自动微分的变量类
 */
enum Nodetype {
    intermediate,
    parameter,
    input,
    reference
};
struct Edge {
    VarPtr parent;
    VarPtr child;
    bool updated;
    Edge(VarPtr parent, VarPtr child, bool updated) : parent(parent), child(child), updated(updated) {}
};
std::ostream& operator<<(std::ostream &os, const Nodetype &type){
    switch(type){
        case intermediate:
            os << "intermediate";
            break;
        case parameter:
            os << "parameter";
            break;
        case input:
            os << "input";
            break;
        default:
            os << "unknown";
    }
    return os;
}
class DataView {
    std::vector<double*> dataview_; // store pointers for reference mode
    std::vector<double> data_;
    bool references_; // true if dataview_ holds pointers, false if it owns data
public:
    using iterator = std::vector<double>::iterator;
    using const_iterator = std::vector<double>::const_iterator;

    class pointer_iterator {
        std::vector<double*>::iterator it_;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = double;
        using difference_type = std::ptrdiff_t;
        using pointer = double*;
        using reference = double&;

        pointer_iterator(std::vector<double*>::iterator it) : it_(it) {}
        reference operator*() const { return **it_; }
        pointer_iterator& operator++() { ++it_; return *this; }
        pointer_iterator operator++(int) { pointer_iterator tmp = *this; ++it_; return tmp; }
        pointer_iterator& operator--() { --it_; return *this; }
        pointer_iterator operator--(int) { pointer_iterator tmp = *this; --it_; return tmp; }
        pointer_iterator operator+(difference_type n) const { return pointer_iterator(it_ + n); }
        pointer_iterator operator-(difference_type n) const { return pointer_iterator(it_ - n); }
        difference_type operator-(const pointer_iterator& other) const { return it_ - other.it_; }
        bool operator==(const pointer_iterator& other) const { return it_ == other.it_; }
        bool operator!=(const pointer_iterator& other) const { return it_ != other.it_; }
    };

    class const_pointer_iterator {
        std::vector<double*>::const_iterator it_;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = double;
        using difference_type = std::ptrdiff_t;
        using pointer = const double*;
        using reference = const double&;

        const_pointer_iterator(std::vector<double*>::const_iterator it) : it_(it) {}
        reference operator*() const { return **it_; }
        const_pointer_iterator& operator++() { ++it_; return *this; }
        const_pointer_iterator operator++(int) { const_pointer_iterator tmp = *this; ++it_; return tmp; }
        const_pointer_iterator& operator--() { --it_; return *this; }
        const_pointer_iterator operator--(int) { const_pointer_iterator tmp = *this; --it_; return tmp; }
        const_pointer_iterator operator+(difference_type n) const { return const_pointer_iterator(it_ + n); }
        const_pointer_iterator operator-(difference_type n) const { return const_pointer_iterator(it_ - n); }
        difference_type operator-(const_pointer_iterator& other) const { return it_ - other.it_; }
        bool operator==(const const_pointer_iterator& other) const { return it_ == other.it_; }
        bool operator!=(const const_pointer_iterator& other) const { return it_ != other.it_; }
    };

    DataView() : references_(false) {}
    // DataView(std::vector<double&>& data) : references_(true) {
    //     for(auto val: data){
    //         dataview_.push_back(&val);
    //     }
    //     // dataview_ = data;
    // }
    DataView(std::vector<double*> &data): references_(true) {
        dataview_ = data;
    }
    DataView(std::vector<double>& data, bool ref=false): references_(ref) {
        if(ref){
            for(auto & val: data){
                dataview_.push_back(&val);
            }
        }else{
            data_ = data;
        }
    }
    DataView(const std::vector<double>& data, bool ref=false): references_(ref) {
        if(ref == true){
            throw std::runtime_error("Cannot create reference DataView from const data");
        }
        // In const mode, always copy
        data_ = data;
    }
    iterator begin() { return references_ ? iterator() : data_.begin(); }
    iterator end() { return references_ ? iterator() : data_.end(); }
    const_iterator begin() const { return references_ ? const_iterator() : data_.begin(); }
    const_iterator end() const { return references_ ? const_iterator() : data_.end(); }
    pointer_iterator ref_begin() { return pointer_iterator(dataview_.begin()); }
    pointer_iterator ref_end() { return pointer_iterator(dataview_.end()); }
    const_pointer_iterator ref_begin() const { return const_pointer_iterator(dataview_.begin()); }
    const_pointer_iterator ref_end() const { return const_pointer_iterator(dataview_.end()); }
    inline size_t size() const { return references_ ? dataview_.size() : data_.size(); }
    inline double& operator[](size_t idx) { return references_ ? *dataview_[idx] : data_[idx]; }
    inline double operator[](size_t idx) const { return references_ ? *dataview_[idx] : data_[idx]; }
    inline bool isref() const { return references_; }
    const std::vector<double> copy() const{
        std::vector<double> new_data;
        if(references_){
            for(auto ptr: dataview_){
                new_data.push_back(*ptr);
            }

        }else{
            new_data = data_;
        }
        return new_data;
    }
};

#include "utils.hpp"

class Variable
{
private:
    // mutable std::vector<double> accumulated_grad_;
    // DataView accumulated_grad_;
    mutable bool updated_ = false;

    DataView data_;
    // std::vector<double> data_;
    // std::vector<double> grad_;
    DataView grad_;
    Nodetype type_;
    
    std::function<void(const DataView&)> grad_fn_;
    // std::function<void(const std::vector<double> &)> grad_fn_;
    std::function<void()> forward_fn_; 
    std::vector<Edge*> children_;
    std::vector<Edge*> parents_;
    std::vector<size_t> shape_;
    public:
    std::vector<VarPtr> ref ; // for reference mode
    std::string name;
    std::string operator_name;
    bool has_grad() const { 
        if(type_ == parameter || type_ == intermediate){
            return true;
        }else if (type_== reference){
            return ref[0]->has_grad(); // depend on referenced ptr
        }else{
            return false;
        }
        // return type_ == parameter || type_ == intermediate; 
    }
    /**
     * 构造函数
     * @param data 数值数据
     * @param type 节点类型
     * @param shape 张量形状（可选）
     */
    explicit Variable() = default;
    static VarPtr Ref_Variable(std::vector<VarPtr> original, std::vector<double*> &data,std::vector<double*> &grad, const std::vector<size_t> &shape = {}){
        auto ret = std::make_shared<Variable>(original, data,grad, reference, shape);
        return ret;
    }
    explicit Variable(std::vector<VarPtr> original, std::vector<double*> &data,std::vector<double*> &grad, Nodetype type=reference, const std::vector<size_t> &shape = {})
        : ref(original),data_(data), grad_(grad), type_(type), shape_(shape)
    {
        if (has_grad())
        {
            if(grad_.size() != data_.size()){
                throw std::runtime_error("Gradient size does not match data size in reference mode");
            }
        }
        if (shape_.empty())
        {
            shape_ = {data_.size()};
        }
        check_validity();
    }
     /**
     * 标量构造函数
     * @param value 数值
     * @param type 节点类型
     * @param shape 张量形状（可选）
     */
    explicit Variable(double value, Nodetype type, const std::vector<size_t> &shape = {})
        : data_({value}, false), type_(type), shape_(shape)
    {
        if (type_ == parameter || type_ == intermediate)
        {
            grad_ = DataView(std::vector<double>(1, 0.0), false);
            // accumulated_grad_ = DataView(std::vector<double>(1, 0.0), false);
        }
        if (shape_.empty())
        {
            shape_ = {data_.size()};
        }
        check_validity();
    }

    /**
     * 向量构造函数
     * @param data 数值向量
     * @param type 节点类型
     * @param shape 张量形状（可选）
     */
    explicit Variable(std::vector<double> &data, Nodetype type, const std::vector<size_t> &shape = {})
        : data_(data), type_(type), shape_(shape)
    {
        if (has_grad())
        {
            grad_ = std::vector<double>(data.size(), 0.0);
            // accumulated_grad_ = std::vector<double>(data.size(), 0.0);
        }
        if (shape_.empty())
        {
            shape_ = {data_.size()};
        }
        check_validity();
    }
    void check_validity()
    {
        size_t total = 1;
        for (size_t idx = 0; idx < ndim(); ++idx)
        {
            size_t dim = shape_[idx];
            if (dim <= 0)
            {
                throw std::runtime_error("Invalid shape dimension" + std::to_string(idx) + ": " + std::to_string(dim));
            }
            total *= dim;
        }
        if (total != data_.size())
        {
            throw std::runtime_error("Data size does not match shape");
        }
        if(type() == reference ){
            if(ref.empty())
                throw std::runtime_error("Reference variable must have at least one referenced variable");
            bool hasgrad = ref[0]->has_grad();
            for(auto & r: ref){
                if(r->has_grad() != hasgrad){
                    throw std::runtime_error("All referenced variables must have the same grad requirement");
                }
            }
        }
    }
    // check if it is some simple shapes like scalar, vector and matrix
    int ndim() const
    {
        return shape_.size();
    }
    bool is_scalar() const
    {
        return size() == 1;
    }
    bool is_vector() const
    {
        return ndim() == 1 && shape_[0] > 1;
    }
    bool is_matrix() const
    {
        return ndim() == 2 && shape_[0] > 1 && shape_[1] > 1;
    }
    // 访问器
    const DataView& data() const { return data_; }
    const DataView& grad() const { return grad_; }
    inline std::vector<Edge*>& children() { return children_; }
    inline std::vector<Edge*>& parents() { return parents_; }
    Nodetype type() const { return type_; }
    size_t size() const { return data_.size(); }
    std::vector<size_t> shape() const { return shape_; }

    // update the data using gradient and learning rate
    void update(double learning_rate)
    {
        for (size_t i = 0; i < data_.size(); i++)
        {
            data_[i] -= learning_rate * grad_[i];
        }
    }
    // 获取标量值（仅适用于单元素）
    double item() const
    {
        if (!is_scalar())
        {
            throw std::runtime_error("item() can only be called on scalar variables");
        }
        return data_[0];
    }

    // 获取梯度标量值（仅适用于单元素）
    double grad_item() const
    {
        if (!has_grad() || !is_scalar())
        {
            throw std::runtime_error("grad_item() can only be called on scalar variables with gradients");
        }
        return grad_[0];
    }
    void set_input(const double new_data){
        if(type_ != input){
            throw std::runtime_error("Only input variable can set data");
        }
        if(data_.size() != 1){
            throw std::runtime_error("This variable is not a scalar");
        }
        assert(!data_.isref());
        data_[0] = new_data;
    }
    void set_input(const std::vector<double>& new_data){
        if(type_ != input){
            throw std::runtime_error("Only input variable can set data");
        }
        if(new_data.size() != data_.size()){
            throw std::runtime_error("New data size does not match");
        }
        assert(!data_.isref());
        for(size_t i = 0; i < new_data.size(); i++)
            data_[i] = new_data[i];
    }
    /**
     * 设置梯度函数
     */
    void set_grad_fn(std::function<void(const DataView&)> grad_fn)
    {
        grad_fn_ = grad_fn;
    }
    void set_forward_fn(std::function<void()> forward_fn)
    {
        forward_fn_ = forward_fn;
    }
    /**
     * 添加子节点
     */
    void add_child(Edge* edge)
    {
        children_.push_back(edge);
    }
    
    void add_parent(Edge* edge)
    {
        parents_.push_back(edge);
    }
    /**
     * 前向计算
     */
    void calc(){
        if(type_ != intermediate && type_ != reference){
            updated_ = true;
            return;
            throw std::runtime_error("Only intermediate nodes can be calculated");
        }else if (type_ == reference){
            // for reference node, just calc its referenced nodes
            for(auto & r: ref){
                if(!r->updated()){
                    r->calc();
                }
            }
            updated_ = true;
            return;
        }
        for(auto & edge: children_){
            if(!edge->child->updated()){
                edge->child->calc();
            }
        }
        if(this->forward_fn_){
            updated_ = true;
            this->forward_fn_();
            double s=0;
            for(int i=0; i< data_.size(); i++){
                s += data_[i];
            }
            // if( s > -1e-6 && s < 1e-6){
            //     std::cout<<"Warning: Forward result is very small: " << s << std::endl;
            //     std::cout<<"variable: " << name << ", op: " << operator_name << std::endl;
            //     std::cout<<"shape:"; print_vec(std::cout, shape_); std::cout<<std::endl;
            // }

        }
    }
    bool updated() const { return updated_; }
    // 累积梯度（不立即传播）
    void accumulate_gradient(const std::vector<double>& grad_input,bool accumulate = true) {
        if (!has_grad()) return;

        // 累积梯度
        if(accumulate){
            double sum_ = 0;
            for (size_t i = 0; i < grad_.size() && i < grad_input.size(); ++i) {
                grad_[i] += grad_input[i];
                sum_ += grad_input[i];
            }
            // if(sum_ < 1e-6 && sum_ > -1e-6){
            //     std::cout<<"Warning: Accumulated gradient is very small: " << sum_ << std::endl;
            //     std::cout<<"variable: " << name << ", op: " << operator_name << std::endl;
            //     std::cout<<"shape:"; print_vec(std::cout, shape_); std::cout<<std::endl;
            // }
        }


        // 检查是否所有父节点都已发送梯度
        bool all_gradients_received = true;
        for (const auto& edge : parents()) {
            if (edge->parent && edge->parent->has_grad()) {
                // 如果父节点需要梯度，检查是否已经从该父节点接收到梯度
                if (edge->updated == false) {
                    all_gradients_received = false;
                    break;
                }
            }
        }
        // 如果所有梯度都已接收，执行反向传播
        if (all_gradients_received) {
            if (grad_fn_) {
                grad_fn_(grad_);
            }
        }
    }
    /**
     * 反向传播计算梯度
     * @param grad_output 从上游传来的梯度
     */
    void backward(const std::vector<double> &grad_output = {})
    {
        if (!has_grad())
            return;

        // 如果是标量输出且没有指定梯度，设为1 (优化目标)
        if (grad_output.empty() && parents_.empty())
        {
            if(data_.size() != 1){
                throw std::runtime_error("Gradient can only be computed for scalar outputs");
            }
            grad_[0] = 1.0;
            // 如果有梯度函数，继续反向传播
            if (grad_fn_)
            {
                grad_fn_(grad_);
            }
        }else {
            throw std::runtime_error("Cannot backward() non-root nodes. Use accumulate_gradient() instead.");
        }

        
    }

    /**
     * 清零梯度
     */
    void zero_grad()
    {
        if (has_grad())
        {
            std::fill(grad_.begin(), grad_.end(), 0.0);
            updated_ = false;
        }
    }
    // void forward(){
    //     if(type_ == intermediate && forward_fn_){
    //         forward_fn_();
    //     }
    // }

    void zero_grad_recursive()
    {
        zero_grad();
        for (auto &edge : children_)
        {
            edge->updated = false;
            if (edge->child != nullptr)
                edge->child->zero_grad_recursive();
        }
    }
    /**
     * 打印变量信息
     */
    std::ostream& operator<<(std::ostream &os) 
    {
        return print(os, false);
    }
    std::ostream& print(std::ostream& os=std::cout, bool verbose=false) 
    {
        os << name<<" = ";
        switch(type_){
            case intermediate:
                os << "intermediate( ";
                break;
            case parameter:
                os << "parameter( ";
                break;
            case input:
                os << "input( ";
                break;
            case reference:
                os << "ref( ";
                break;
            default:
                throw std::runtime_error("Unknown variable type");
        }
        os << "size=" << size() << ", ";
        os << "shape=";
        print_vec(os, shape_); os<<", ";
        if(type_ == intermediate && !operator_name.empty()){
            os << "op=" << operator_name;
            print_vec(os,children_, '(',',',')', [](std::ostream& os, Edge* edge){ os << edge->child->name;});
            os<<", ";
        }
        if(verbose){
            os<< "data=";
            print_vec(os, data_);
            os<<", ";
        }

        if (has_grad() && verbose)
        {
            os << "grad=";
            print_vec(os, grad_, "[", ",", "]");
        }
        os << ")" << std::endl;
        return os;
    }
    // 通过扁平索引获取多维索引
    std::vector<int> PlainItemIndex(size_t flat_index)
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        std::vector<int> idx(ndim());
        size_t temp = flat_index;
        for (int i = ndim() - 1; i >= 0; i--)
        {
            idx[i] = temp % shape_[i];
            temp /= shape_[i];
        }
        return idx;
    }
    // 索引访问函数，支持多维索引
    size_t ItemIndex(const std::vector<int> &idx)
    {
        if (idx.size() != shape().size())
        { // dimension match
            throw std::runtime_error("Index dimension does not match variable dimension");
        }
        for (size_t i = 0; i < idx.size(); i++)
        {
            if (idx[i] < 0 || static_cast<size_t>(idx[i]) >= shape()[i])
            {
                throw std::runtime_error("Index out of bounds at dimension " + std::to_string(i));
            }
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < idx.size(); i++)
        {
            size_t stride = 1;
            for (size_t j = i + 1; j < shape().size(); j++)
            {
                stride *= shape()[j];
            }
            flat_index += idx[i] * stride;
        }
        return flat_index;
    }
    // size_t ItemIndex(const std::vector<size_t> &idx)
    // {
    //     if (idx.size() != shape().size())
    //     { // dimension match
    //         throw std::runtime_error("Index dimension does not match variable dimension");
    //     }
    //     return ItemIndex(std::vector<int>(idx.begin(), idx.end()));
    // }

    // 通过多维索引获取元素
    double* ItemAddr(const std::vector<int> &idx)
    {
        return &data_[ItemIndex(idx)];
    }
    double* ItemAddr(size_t flat_index)
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return &data_[flat_index];
    }
    double& Item(const std::vector<int> &idx)
    {
        return data_[ItemIndex(idx)];
    }
    double& Item(size_t flat_index)
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return data_[flat_index];
    }
    double* GradItemAddr(const std::vector<int> &idx)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        return &grad_[ItemIndex(idx)];
    }
    double* GradItemAddr(size_t flat_index)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return &grad_[flat_index];
    }
    // 通过多维索引获取梯度
    double& GradItem(const std::vector<int> &idx)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        return grad_[ItemIndex(idx)];
    }
    double& GradItem(size_t flat_index)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return grad_[flat_index];
    }

};

// 工具函数：创建Variable的智能指针

static int var_counter = 0;
static int param_counter = 0;
static int input_counter = 0;
static int ref_counter = 0;
VarPtr make_var(double value)
{
    VarPtr a = std::make_shared<Variable>(value, intermediate);
    a->name = "var" + std::to_string(var_counter++);
    return a;
}

VarPtr make_var(std::vector<double> data, const std::vector<size_t> &shape = {})
{
    VarPtr a = std::make_shared<Variable>(data, intermediate, shape);
    a->name = "var" + std::to_string(var_counter++);
    return a;
}


VarPtr make_ref(VarPtr var,std::vector<double*> &data,std::vector<double*> grad, const std::vector<size_t> &shape = {})
{
    auto a = Variable::Ref_Variable({var}, data, grad, shape);
    a->name = "ref_" + var->name + "_" + std::to_string(ref_counter++);
    return a;
}
VarPtr make_ref(std::vector<VarPtr> vars,std::vector<double*> &data,std::vector<double*> grad, const std::vector<size_t> &shape = {})
{
    auto a = Variable::Ref_Variable(vars, data, grad, shape);
    a->name = "ref_multi_";
    for(auto var: vars){
        a->name += var->name + "_";
    }
    a->name += std::to_string(ref_counter++);
    return a;
}

VarPtr make_param(double value, const std::vector<size_t> &shape = {})
{
    VarPtr a = std::make_shared<Variable>(value, parameter, shape);
    a->name = "param" + std::to_string(param_counter++);
    return a;
}
VarPtr make_param( std::vector<double> data, const std::vector<size_t> &shape = {})
{
    VarPtr a = std::make_shared<Variable>(data, parameter, shape);
    a->name = "param" + std::to_string(param_counter++);
    return a;
}
VarPtr make_input(double value, const std::vector<size_t> &shape = {})
{
    VarPtr a = std::make_shared<Variable>(value, input, shape);
    a->name = "input" + std::to_string(input_counter++);
    return a;
}
VarPtr make_input( std::vector<double> data, const std::vector<size_t> &shape = {})
{
    VarPtr a = std::make_shared<Variable>(data, input, shape);
    a->name = "input" + std::to_string(input_counter++);
    return a;
}

#include "operations.hpp"

VarPtr operator+(VarPtr a, VarPtr b) { return add(a, b); }
VarPtr operator-(VarPtr a, VarPtr b) { return sub(a, b); }
VarPtr operator*(VarPtr a, VarPtr b) { return mul(a, b); }
// VarPtr operator/(VarPtr a, VarPtr b) { return div(a, b); }  // div function is commented out
VarPtr operator^(VarPtr a, double exponent) { return pow_elementwise(a, exponent); }


class ComputationGraph {
    public:
    std::vector<VarPtr> input_nodes;
    std::vector<VarPtr> parameter_nodes;
    std::vector<VarPtr> intermediate_nodes;
    std::vector<VarPtr> reference_nodes;
    std::vector<VarPtr> output_nodes; // only 1 output node in most cases
    ComputationGraph() = default;
    ComputationGraph(std::vector<VarPtr> inputs,
                     std::vector<VarPtr> parameters,
                     std::vector<VarPtr> intermediates,
                     std::vector<VarPtr> references,
                     std::vector<VarPtr> outputs)
        : input_nodes(std::move(inputs)),
            reference_nodes(std::move(references)),
          parameter_nodes(std::move(parameters)),
          intermediate_nodes(std::move(intermediates)),
          output_nodes(std::move(outputs)) {}
    static ComputationGraph BuildFromOutput(VarPtr output_node){
        ComputationGraph graph;
        std::vector<VarPtr> stack;
        std::unordered_set<VarPtr> visited;
        stack.push_back(output_node);
        while(!stack.empty()){
            VarPtr current = stack.back();
            stack.pop_back();
            if(visited.find(current) != visited.end()){
                continue;
            }
            visited.insert(current);
            switch(current->type()){
                case input:
                    graph.input_nodes.push_back(current);
                    break;
                case parameter:
                    graph.parameter_nodes.push_back(current);
                    break;
                case intermediate:
                    graph.intermediate_nodes.push_back(current);
                    break;
                case reference:
                    graph.reference_nodes.push_back(current);
                    break;
                default:
                    throw std::runtime_error("Unknown node type in computation graph");
            }
            for(auto & edge: current->children()){
                if(edge->child && visited.find(edge->child) == visited.end()){
                    stack.push_back(edge->child);
                }
            }
        }
        return graph;
    }
    static std::vector<VarPtr> toposort(ComputationGraph &graph){
        std::vector<VarPtr> sorted_nodes;
        std::vector<VarPtr> stack;
        std::map<VarPtr, bool> visited;
        for(auto & node: graph.input_nodes){
            stack.push_back(node);
        }
        for(auto & node: graph.parameter_nodes){
            stack.push_back(node);
        }
        while(!stack.empty()){
            VarPtr current = stack.back();
            sorted_nodes.push_back(current);
            visited[current] = true;
            stack.pop_back();
            for(auto & edge: current->parents()){
                auto parent = edge->parent;
                if(!parent) continue;
                if(visited.find(parent) != visited.end()){
                    continue;
                }
                bool flag = all_of(parent->children().begin(), parent->children().end(), [&](const auto& e) {
                    return visited.find(e->child) != visited.end();
                });
                if(flag){
                    stack.push_back(parent);
                }
            }
        }
        return sorted_nodes;
    }

    void SaveParams(string filename) {
        std::ofstream ofs(filename);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for saving parameters");
        }
        for (const auto& param : parameter_nodes) {
            ofs << param->name << ": ";
            print_vec(ofs, param->data(), "",",", "");
            ofs << std::endl;
        }
    }
    void LoadParams(string filename) {
        std::ifstream ifs(filename);
        if (!ifs) {
            throw std::runtime_error("Failed to open file for loading parameters");
        }
        std::string line;
        while (std::getline(ifs, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) {
                continue; // Skip invalid lines
            }
            std::string name = line.substr(0, colon_pos);
            std::string values_str = line.substr(colon_pos + 1);
            std::vector<double> values;
            size_t start = 0;
            size_t end = values_str.find(',');
            while (end != std::string::npos) {
                values.push_back(std::stod(values_str.substr(start, end - start)));
                start = end + 1;
                end = values_str.find(',', start);
            }
            // Add the last value
            if (start < values_str.size()) {
                values.push_back(std::stod(values_str.substr(start)));
            }
            // Find the parameter by name and update its data
            for (auto& param : parameter_nodes) {
                if (param->name == name) {
                    if (param->data().size() != values.size()) {
                        throw std::runtime_error("Parameter size mismatch for " + name);
                    }
                    for (size_t i = 0; i < values.size(); ++i) {
                        param->Item(i) = values[i];
                    }
                    break;
                }
            }
        }
    }
    void SaveArch(string filename){
        std::ofstream ofs(filename);
        std::vector<VarPtr> sorted = toposort(*this);
        for(auto & node: sorted){
            node->print(ofs,false);
        }
    }
};