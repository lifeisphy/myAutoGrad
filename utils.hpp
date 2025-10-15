#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#pragma once


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
               char open_bracket, char delimiter, char close_bracket, Func func) {
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