#include <cstddef>
#include<vector>

#include "tensor.hpp"

using namespace std;

// Constructor : initialize
Tensor::Tensor(const vector<size_t>& input_shape) {
    // shape
    this->shape_ = input_shape;
    size_t num_elements = 1;

    for(auto const& s: this->shape_) {
        num_elements *= s;
    }

    // data
    vector<float> elements;
    for(size_t i = 0; i < num_elements; i++){
        elements.push_back(0.0f);
    }
    this->data_ = elements;
}

// numel : return data size
size_t Tensor::numel() const {
    return this->data_.size();
}

// shape() : return shape
const vector<size_t>& Tensor::shape() const {
    return this->shape_;    
}

// fill : fill the vlaue to all the data in data_
void Tensor::fill(float value) {
    for(size_t i = 0; i < this->data_.size(); i++) {
        this->data_[i] = value;
    }
}

// at : return data[index]
float Tensor::at(size_t index) const {
    return this->data_.at(index);
}

// set : set value to data[idnex]
void Tensor::set(size_t index, float value) {
    this->data_.at(index) = value;
}




