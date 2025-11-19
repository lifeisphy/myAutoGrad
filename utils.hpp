#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
std::string to_string(const Nodetype &type){
    switch(type){
        case intermediate:
            return "intermediate";
        case parameter:
            return "parameter";
        case input:
            return "input";
        default:
            return "unknown";
    }
}
std::ostream& operator<<(std::ostream &os, const Nodetype &type){
    os << std::to_string(type);
    return os;
}


// 工具函数：创建Variable的智能指针
std::map<std::string, int> counter;
std::string get_name(const std::string &prefix, const std::string default_name){
    auto name_pref = prefix.empty()? default_name : prefix;
    if(counter.find(name_pref) == counter.end()){
        counter[name_pref] = 1;
        return name_pref;
    }else {
        return name_pref + std::to_string(counter[name_pref]++);
    }
}

std::ostream& operator<<(std::ostream& os, const idx_range& r){
    os << std::get<0>(r) << ":" << std::get<1>(r) << ":" << std::get<2>(r);
    return os;
}
// 简单版本的 print_vec（2个参数）
template<typename T>
void print_vec(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        os << vec[i];
    }
    os << "]";
}

// 带自定义分隔符的 print_vec（5个参数）
template<typename T>
void print_vec(std::ostream& os, const std::vector<T>& vec, 
               std::string open_bracket, std::string delimiter, std::string close_bracket) {
    os << open_bracket;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << delimiter << " ";
        os << vec[i];
    }
    os << close_bracket;
}
template<typename T, typename Func>
void print_vec(std::ostream& os, const std::vector<T>& vec, 
               std::string open_bracket, std::string delimiter, std::string close_bracket, Func func) {
    os << open_bracket;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << delimiter << " ";
        func(os, vec[i]);
    }
    os << close_bracket;
}

void print_vec(std::ostream& os, const DataView& data_view) {
    os << "[";
    for (size_t i = 0; i < data_view.size(); ++i) {
        if (i > 0) os << ", ";
        os << data_view[i];
    }
    os << "]";
}

void print_vec(std::ostream& os, const DataView& data_view, string open_bracket, string delimiter, string close_bracket) {
    os << open_bracket;
    for (size_t i = 0; i < data_view.size(); ++i) {
        if (i > 0) os << delimiter;
        os << data_view[i];
    }
    os << close_bracket;
}

std::vector<double> vec_r(size_t size, double scale = 0.1){
    std::vector<double> data(size);
    for(size_t i=0; i < size; i++){
        data[i] = (rand() / double(RAND_MAX) - 0.5) * 2.0 * scale;  // [-scale, scale]
    }
    return data;
}

std::vector<double> zero_vec(size_t size){
    return std::vector<double>(size, 0.0);
}

std::vector<int> range(int start, int stop, int step, int size){
    std::vector<int> result;
    if(step ==0) throw std::runtime_error("Slice step cannot be zero.");
    if(start <= - size || start >= size) throw std::runtime_error("Slice start index out of bounds.");
    if(stop <= - size || stop > size) throw std::runtime_error("Slice stop index out of bounds.");
    // std::cout<<"params: "<< start <<","<< stop <<","<< step <<","<< size <<std::endl;
    if(step < 0){
        if(start < 0) start = size + start;
        if(stop < 0 ) {
            if (size + stop  <  start) stop = size + stop;
        }
    }else {
        if(start < 0) start = size + start;
        if(stop < 0 ) stop = size + stop;
    }
    for(int i=start; (step>0) ? (i<stop) : (i>stop);i+=step){
        result.push_back(i);
    }
    return result;
}

std::vector<idx_range> parse_slices(const std::string slice_str, const std::vector<size_t>& shape){
    // slice: start:stop:step, start2:stop2:step2,...
    // 解析切片字符串
    std::vector<idx_range> slices;
    std::istringstream ss(slice_str);
    std::string token;
    size_t dim = 0;
    auto empty = [](std::string s){
        return s.find_first_not_of(" \t\n\r") == std::string::npos;
    };
    while (std::getline(ss, token, ',')) {
        if(dim >= shape.size()){
            throw std::runtime_error("Too many slice dimensions provided.");
        }
        int start = 0;
        int stop = shape[dim];
        int step = 1;

        size_t first_colon = token.find(':');
        size_t second_colon = token.find(':', first_colon + 1);

        std::string substr;
        if (first_colon != std::string::npos) {
            // 有至少一个冒号
            if (first_colon > 0) {
                substr = token.substr(0, first_colon);
                if(!empty(substr)){
                    start = std::stoi(substr);
                }
            }
            if (second_colon != std::string::npos) {
                // 有第二个冒号
                if (second_colon > first_colon + 1) {
                    substr = token.substr(first_colon + 1, second_colon - first_colon - 1);
                    if(!empty(substr)){
                        stop = std::stoi(substr);
                    }
                }

                if (second_colon + 1 < token.size()) {
                    substr = token.substr(second_colon + 1);
                    if(!empty(substr)){
                        step = std::stoi(substr);
                        if(step < 0 && start == 0 && stop == shape[dim]){
                            // 特殊处理::-1
                            start = shape[dim]-1;
                            stop = -1;
                        }
                    }
                }
            } else {
                // 只有一个冒号
                if (first_colon + 1 < token.size()) {
                    substr = token.substr(first_colon + 1);
                    if(!empty(substr)){
                        stop = std::stoi(substr);
                    }
                }
            }
        } else {
            // 没有冒号，单一索引
            if(!empty(token)){
                start = std::stoi(token);
                stop = start == -1 ? shape[dim] : start + 1;
            }
        }

        slices.push_back(std::make_tuple(start, stop, step));
        dim++;
    }
    return slices;
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
