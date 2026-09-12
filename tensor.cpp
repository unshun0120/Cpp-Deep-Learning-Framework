#include<cstddef>
#include<vector>
#include<cassert>
#include<algorithm>

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
    // requires grad
    this->requires_grad_ = false;

    // grad
    vector<float> grad_elements;
    for(size_t i = 0; i < num_elements; i++){
        grad_elements.push_back(0.0f);
    }
    this->grad_ = grad_elements;

    this->op_ = OpType::None;

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
Tensor Tensor::add(Tensor& other) {
    
    assert(this->shape_ == other.shape());

    Tensor result(this->shape_);
    
    // add
    for(size_t i = 0; i < this->numel(); i++) {
        result.set(i, this->at(i) + other.at(i));
    }

    //autograd
    result.op_ = OpType::Add;

    if(this->requires_grad() == true) {
        result.requires_grad_ = true;
    }
    else if(other.requires_grad() == true) {
        result.requires_grad_ = true;
    }

    // parent
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    return result;
}

// multiply
Tensor Tensor::multiply(Tensor& other) {
    assert(this->shape_ == other.shape());

    Tensor result(this->shape_);

    for(size_t i = 0; i < this->numel(); i++) {
        result.set(i, this->at(i) * other.at(i));
    }  

     //autograd
    result.op_ = OpType::Multiply;

    if(this->requires_grad() == true) {
        result.requires_grad_ = true;
    }
    else if(other.requires_grad() == true) {
        result.requires_grad_ = true;
    }

    // parent
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    return result;
}

// 矩陣乘法
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

// requires grad
bool Tensor::requires_grad() const {
    return this->requires_grad_;
}

void Tensor::set_requires_grad(bool value) {
    this->requires_grad_ = value;
}

// grad
float Tensor::grad(size_t index) const {
    return this->grad_.at(index);
}

void Tensor::set_grad(size_t index, float value) {
    this->grad_.at(index) = value;
}

// Getter function (for test)
OpType Tensor::op() const {
    return this->op_;
}

size_t Tensor::num_parents() const {
    return this->parents_.size();
}

const Tensor* Tensor::parent(size_t index) const {
    return this->parents_.at(index);
}

// backward add
// c = a + b
// a.grad[i] += out.grad[i]
// b.grad[i] += out.grad[i]
void Tensor::backward_add() {
    assert(this->op_ == OpType::Add);
    
    for(size_t i = 0; i < this->grad_.size(); i++) {
        for(size_t j = 0; j < this->parents_.size(); j++) {
            this->parents_.at(j)->set_grad(i, this->parents_.at(j)->grad(i) + this->grad(i));
        }  
    }
}

// backward multiply
// c = a * b
// a.grad[i] += out.grad[i] * b[i]
// b.grad[i] += out.grad[i] * a[i]
void Tensor::backward_multiply() {
    assert(this->op_ == OpType::Multiply);
    
    for(size_t i = 0; i < this->grad_.size(); i++) {
        this->parents_.at(0)->set_grad(i, 
            this->parents_.at(0)->grad(i) + this->grad(i) * this->parents_.at(1)->data_[i]);
        
        this->parents_.at(1)->set_grad(i, 
            this->parents_.at(1)->grad(i) + this->grad(i) * this->parents_.at(0)->data_[i]);       
    }
}

// topology (DFS)
vector<Tensor*> Tensor::build_topology() {
    vector<Tensor*> visited;
    vector<Tensor*> order;

    this->traversal(this, visited, order);
    
    return order;
}

// postorder DFS, parent在前面, 當前node在最後面
void Tensor::traversal(Tensor* node, vector<Tensor*>& visited, vector<Tensor*>& order) {
    if(find(visited.begin(), visited.end(), node) != visited.end())
        return;

    visited.push_back(node);

    for(size_t i = 0; i < node->parents_.size(); i++) {
        node->traversal(node->parents_[i], visited, order);
    }

    order.push_back(node);
}

// backward()
void Tensor::backward() {
    assert(this->numel() == 1);
    this->grad_[0] = 1;

    vector<Tensor*> order = this->build_topology();
    
    for(int i = order.size()-1 ; i >= 0; i--) {
        if(order[i]->op_ == OpType::Add) {
            order[i]->backward_add();
        }

        if(order[i]->op_ == OpType::Multiply) {
            order[i] ->backward_multiply();
        }
    }
}











