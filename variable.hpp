#pragma once
#include "autograd.hpp"
class Variable;
using VarPtr = std::shared_ptr<Variable>;
/**
 * 支持自动微分的变量类
 */


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
            // we allow referenced variables to have different grad requirements
            // for(auto & r: ref){
                // if(r->has_grad() != hasgrad){
                //     throw std::runtime_error("All referenced variables must have the same grad requirement");
                // }
            // }
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
    bool require_all_gradients_ = true;
    bool require_all_gradients() const {
        return require_all_gradients_;
    }

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
        std::cout<<"Accumulate gradient in "<< name <<std::endl;
        // 累积梯度
        if(accumulate){
            for (size_t i = 0; i < grad_.size() && i < grad_input.size(); ++i) {
                if(!grad_.is_nullptr(i)){
                    grad_[i] += grad_input[i];
                }
            }
        }

        if(require_all_gradients()){
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
                std::cout<<"All gradients received for "<< name <<", calling grad_fn."<<std::endl;
                if (grad_fn_) {
                    grad_fn_(grad_);
                }
            }else{
                std::cout<<"Not all gradients received for "<< name <<", waiting..."<<std::endl;
                for(auto& edge: parents()){
                    if(edge->parent && edge->parent->has_grad()){
                        std::cout<<"  Parent "<< edge->parent->name <<", updated="<< (edge->updated ? "true":"false") <<std::endl;
                    }
                }
            }
        }else {
            if(grad_fn_) {
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
        std::cout<<"Bakward() in " << name <<std::endl;
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
            case parameter:
                os << "parameter( ";
                os << "size=" << size() << ", ";
                os << "shape=";
                print_vec(os, shape_); os<<", ";
                break;
            case input:
                os << "input( ";
                os << "size=" << size() << ", ";
                os << "shape=";
                print_vec(os, shape_); os<<", ";
                break;
            case intermediate:
                os << operator_name << "(";
                print_vec(os,children_, "",",","", [](std::ostream& os, Edge* edge){ os << edge->child->name;});
                os<<", ";
                break;
            case reference:
                if(!operator_name.empty()){
                    os<<operator_name<<"(";
                }else{
                    os<<"ref(";
                }
                print_vec(os, ref, "[",",","]", [](std::ostream& os, VarPtr v){ os << v->name;});
                os <<", ";
                break;
            default:
                throw std::runtime_error("Unknown variable type");
        }
        os<<"updated="<< (updated_ ? "true":"false") <<", ";
        os<<"require_all_gradients="<< (require_all_gradients_ ? "true":"false") <<", ";
        if(verbose){
            os<< "data=";
            print_vec(os, data_);
            os<<", ";
            if (has_grad())
            {
                os << "grad=";
                print_vec(os, grad_, "[", ",", "]");
            }
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
