#pragma once
#include <vector>
#include <iterator>
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
    bool is_nullptr(size_t idx) const{
        if(references_){
            return dataview_[idx] == nullptr;
        }else{
            return false;
            // throw std::runtime_error("is_nullptr called on non-reference DataView");
        }
    }
    inline double& operator[](size_t idx) {
        return references_ ? *dataview_[idx] : data_[idx]; }
    inline double operator[](size_t idx) const {
        return references_ ? *dataview_[idx] : data_[idx]; }
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

