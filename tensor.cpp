#include<cstddef>
#include<vector>
#include<cassert>

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

// at (2D)
float Tensor::at(size_t row, size_t col) const {  
    assert(shape_.size() == 2);
    assert(row < shape_[0]);
    assert(col < shape_[1]);

    return this->data_.at(row * shape_[1] + col);
}

// set (2D)
void Tensor::set(size_t row, size_t col, float value) {
    assert(shape_.size() == 2);
    assert(row < shape_[0]);
    assert(col < shape_[1]);

    this->data_.at(row * shape_[1] + col) = value;
}

// reshape
void Tensor::reshape(const vector<size_t>& new_shape) {
    size_t num_elements = 1;
    for(auto const& dim : new_shape) {
        num_elements *= dim;
    }
    assert(num_elements == this->numel());
    this->shape_ = new_shape;
}

// add
Tensor Tensor::add(const Tensor& other) const {
    
    assert(this->shape_ == other.shape());

    Tensor result(this->shape_);

    for(size_t i = 0; i < this->numel(); i++) {
        result.set(i, this->at(i) + other.at(i));
    }  

    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    assert(this->shape_ == other.shape());

    Tensor result(this->shape_);

    for(size_t i = 0; i < this->numel(); i++) {
        result.set(i, this->at(i) * other.at(i));
    }  

    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(this->shape_.size() == 2);
    assert(other.shape_.size() == 2);
    assert(this->shape_[1] == other.shape_[0]);

    Tensor result({this->shape_[0], other.shape_[1]});   

    for(size_t i = 0; i < this->shape_[0]; i++) {
        for(size_t j = 0; j < other.shape_[1]; j++) {
            float sum = 0;
            for(size_t k = 0; k < this->shape_[1]; k++) {
                sum += this->at(i, k) * other.at(k, j);  
            }
            result.set(i, j, sum);
        }
    }
    
    return result;
}




















