#include <cstddef>
#include<vector>

class Tensor {
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;   
    
public:
    // constructor
    Tensor(const std::vector<size_t>& input_shape);
    
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
    
    Tensor add(const Tensor& other) const;
};


