
#pragma once
#include "autograd.hpp"
#include <iomanip>
#include <sstream>
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
        graph.output_nodes.push_back(output_node);
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
    VarPtr get_node_by_name(const string& name){
        for(auto & node: input_nodes){
            if(node->name == name){
                return node;
            }
        }
        for(auto & node: parameter_nodes){
            if(node->name == name){
                return node;
            }
        }
        for(auto & node: intermediate_nodes){
            if(node->name == name){
                return node;
            }
        }
        for(auto & node: reference_nodes){
            if(node->name == name){
                return node;
            }
        }
        throw std::runtime_error("Node with name " + name + " not found in computation graph");
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
            auto node = get_node_by_name(name);
            if(node->type() != parameter){
                throw std::runtime_error("Node " + name + " is not a parameter node");
            }

            std::string values_str = line.substr(colon_pos + 1);
            
            size_t start = 0;
            size_t end = values_str.find(',');
            int i=0;
            while (end != std::string::npos) {
                node->Item(i++) = std::stod(values_str.substr(start, end - start));
                start = end + 1;
                end = values_str.find(',', start);

            }
            
            // Add the last value
            if (start < values_str.size()) {
                node->Item(i++) = std::stod(values_str.substr(start));
            }
            if(i != node->size()){
                throw std::runtime_error("Loaded parameter size does not match for node " + name);
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
    int epochs = 0;
    int epoch = 0;
    int n_samples = 0;
    int i = 0;
    void fit(std::function<void(ComputationGraph* pgraph)> load_data, int epochs, int n_samples, double learning_rate, std::function<void(ComputationGraph* pgraph)> print_info_before=nullptr, std::function<void(ComputationGraph* pgraph)> print_info_after=nullptr){
        auto optimizer = AdamOptimizer(learning_rate);
        optimizer.set_parameter_nodes(parameter_nodes);
        this->n_samples = n_samples;
        this->epochs = epochs;
        for(int epoch = 0; epoch < epochs; epoch++){
            this->epoch = epoch;
            for(int i = 0; i < n_samples; i++){
                this->i = i;
                load_data(this);
                if(output_nodes.empty() || output_nodes.size() != 1){
                    throw std::runtime_error("No output node or many output nodes in computation graph");
                }
                output_nodes[0]->zero_grad_recursive();
                output_nodes[0]->calc();
                if(print_info_before){
                    print_info_before(this);
                }
                output_nodes[0]->backward();
                for(auto & param: parameter_nodes){
                    optimizer.update(param);
                    // param->update(learning_rate);
                }
                if(print_info_after){
                    print_info_after(this);
                }
            }
        }
    }
    void print_summary(){
        auto print_nodes = [](std::vector<VarPtr>& nodes, const std::string& title){
            if (nodes.empty()) {
                std::cout << title << " (0): None" << std::endl << std::endl;
                return;
            }
            
            std::cout << title << " (" << nodes.size() << "), ";
            
            // 计算各列的最大宽度
            size_t max_name_width = 4;  // "Name" 的长度
            size_t max_type_width = 4;  // "Type" 的长度
            size_t max_shape_width = 5; // "Shape" 的长度
            int dim = 0;
            for(auto & node: nodes){
                max_name_width = std::max(max_name_width, node->name.length());
                max_type_width = std::max(max_type_width, to_string(node->type()).length());
                dim += node->size();
                // 计算 shape 字符串的长度
                std::stringstream shape_ss;
                print_vec(shape_ss, node->shape(), "[", ",", "]");
                max_shape_width = std::max(max_shape_width, shape_ss.str().length());
            }
            std::cout << "Total dimension: " << dim << std::endl;
            // 打印表头
            std::cout << std::left 
                      << std::setw(max_name_width + 2) << "Name"
                      << std::setw(max_type_width + 2) << "Type"
                      << std::setw(max_shape_width + 2) << "Shape"
                      << "Data" << std::endl;
            
            // 打印分隔线
            std::cout << std::string(max_name_width + 2, '-')
                      << std::string(max_type_width + 2, '-')
                      << std::string(max_shape_width + 2, '-')
                      << std::string(10, '-') << std::endl;
            
            // 打印数据行
            for(auto & node: nodes){
                std::stringstream shape_ss;
                print_vec(shape_ss, node->shape(), "[", ",", "]");
                
                std::cout << std::left 
                          << std::setw(max_name_width + 2) << node->name
                          << std::setw(max_type_width + 2) << to_string(node->type())
                          << std::setw(max_shape_width + 2) << shape_ss.str();
                
                // 打印数据（如果数据太长，只显示前几个元素）
                node->print();
            }
            std::cout << std::endl;
        };
        std::cout << "Computation Graph Summary:" << std::endl;
        std::cout << "Input Nodes: " << input_nodes.size() << std::endl;
        print_nodes(input_nodes, "Input Nodes");
        print_nodes(parameter_nodes, "Parameter Nodes");
        print_nodes(intermediate_nodes, "Intermediate Nodes");
        print_nodes(reference_nodes, "Reference Nodes");
        print_nodes(output_nodes, "Output Nodes");
    }
};
