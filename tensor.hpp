#pragma once

#include<cstddef>
#include<vector>
 
// 記這個tensor是從哪些東西得到的
// operation type
enum class OpType {
    None,
    Add
};

class Tensor {
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;   
    
    bool requires_grad_;
    std::vector<float> grad_;
    
    OpType op_;
    std::vector<Tensor*> parents_;

public:
    // constructor
    Tensor(const std::vector<size_t>& input_shape);
    
    // characteristic
    // 在function後面寫const, 告訴說這個function不會去改到Tensor物件本身的東西
    size_t numel() const;
   
    // 前面加上const表示拿到這個reference的人不可以修改
    const std::vector<size_t>& shape() const;

    void fill(float value);
    
    float at(size_t index) const; 

    void set(size_t index, float value);

    float at(size_t row, size_t col) const;
    
    void set(size_t row, size_t col, float value);
    
    void reshape(const std::vector<size_t>& new_shape);
    
    // operation
    Tensor add(Tensor& other);

    Tensor multiply(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;
    
    // Autograd
    bool requires_grad() const;
    
    void set_requires_grad(bool value);

    float grad(size_t index) const;
    
    void set_grad(size_t index, float value);
    
    void backward_add();


    // For test
    // 單純用來測試, 拿到tensor的資訊
    OpType op() const;
    size_t num_parents() const;
    const Tensor* parent(size_t index) const;

};

