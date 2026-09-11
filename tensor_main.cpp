#include <cassert>

#include "tensor.hpp"

int main() {
    // Constructor, numel
    Tensor numel_a({2, 3});
    Tensor numel_b({2, 3, 4});

    assert(numel_a.numel() == 6);
    assert(numel_b.numel() == 24);

    // shape
    Tensor shape_t({2, 3});

    assert(shape_t.shape().size() == 2);
    assert(shape_t.shape()[0] == 2);
    assert(shape_t.shape()[1] == 3);

    // fill, at
    Tensor fill_t({2, 3});

    fill_t.fill(5.0f);

    for(size_t i = 0; i < fill_t.numel(); i++) {
        assert(fill_t.at(i) == 5.0f);
    }

    fill_t.fill(2.5f);

    for(size_t i = 0; i < fill_t.numel(); i++) {
        assert(fill_t.at(i) == 2.5f);
    }

    // set
    Tensor set_t({2, 3});

    set_t.fill(1.0f);
    set_t.set(2, 10.0f);

    for(size_t i = 0; i < set_t.numel(); i++) {
        if(i == 2)
            assert(set_t.at(i) == 10.0f);
        else
            assert(set_t.at(i) == 1.0f);
    }

    // 2D at
    Tensor index_t({2, 3});

    index_t.fill(0.0f);

    index_t.set(0, 0, 1.0f);
    index_t.set(0, 2, 2.0f);
    index_t.set(1, 0, 3.0f);
    index_t.set(1, 2, 4.0f);

    assert(index_t.at(0, 0) == 1.0f);
    assert(index_t.at(0, 2) == 2.0f);
    assert(index_t.at(1, 0) == 3.0f);
    assert(index_t.at(1, 2) == 4.0f);

    assert(index_t.at(0, 0) == index_t.at(0));
    assert(index_t.at(0, 2) == index_t.at(2));
    assert(index_t.at(1, 0) == index_t.at(3));
    assert(index_t.at(1, 2) == index_t.at(5));

    // reshape
    Tensor reshape_t({2, 3});

    for(size_t i = 0; i < reshape_t.numel(); i++) {
        reshape_t.set(i, static_cast<float>(i));
    }

    reshape_t.reshape({3, 2});

    assert(reshape_t.shape().size() == 2);
    assert(reshape_t.shape()[0] == 3);
    assert(reshape_t.shape()[1] == 2);
    assert(reshape_t.numel() == 6);

    assert(reshape_t.at(0) == 0.0f);
    assert(reshape_t.at(1) == 1.0f);
    assert(reshape_t.at(2) == 2.0f);
    assert(reshape_t.at(3) == 3.0f);
    assert(reshape_t.at(4) == 4.0f);
    assert(reshape_t.at(5) == 5.0f);

    assert(reshape_t.at(2, 1) == 5.0f);

    reshape_t.reshape({6});
    assert(reshape_t.shape().size() == 1);
    assert(reshape_t.shape()[0] == 6);
    assert(reshape_t.numel() == 6);

    reshape_t.reshape({1, 2, 3});
    assert(reshape_t.shape().size() == 3);
    assert(reshape_t.numel() == 6);

    // add
    Tensor add_a({2, 3});
    Tensor add_b({2, 3});

    for(size_t i = 0; i < 6; i++) {
        add_a.set(i, static_cast<float>(i + 1));
        add_b.set(i, static_cast<float>((i + 1) * 10));
    }

    Tensor add_c = add_a.add(add_b);

    assert(add_c.shape() == add_a.shape());
    assert(add_c.numel() == 6);

    assert(add_c.at(0) == 11.0f);
    assert(add_c.at(1) == 22.0f);
    assert(add_c.at(2) == 33.0f);
    assert(add_c.at(3) == 44.0f);
    assert(add_c.at(4) == 55.0f);
    assert(add_c.at(5) == 66.0f);

    assert(add_a.at(0) == 1.0f);
    assert(add_b.at(0) == 10.0f);

    // multiply
    Tensor multiply_a({2, 3});
    Tensor multiply_b({2, 3});

    for(size_t i = 0; i < 6; i++) {
        multiply_a.set(i, static_cast<float>(i + 1));
        multiply_b.set(i, static_cast<float>(i + 2));
    }

    Tensor multiply_c = multiply_a.multiply(multiply_b);

    assert(multiply_c.shape() == multiply_a.shape());
    assert(multiply_c.numel() == 6);

    assert(multiply_c.at(0) == 2.0f);
    assert(multiply_c.at(1) == 6.0f);
    assert(multiply_c.at(2) == 12.0f);
    assert(multiply_c.at(3) == 20.0f);
    assert(multiply_c.at(4) == 30.0f);
    assert(multiply_c.at(5) == 42.0f);

    assert(multiply_a.at(0) == 1.0f);
    assert(multiply_b.at(0) == 2.0f);

    // 2D matmul 
    Tensor matmul_a({2, 3});
    Tensor matmul_b({3, 2});

    matmul_a.set(0, 0, 1.0f);
    matmul_a.set(0, 1, 2.0f);
    matmul_a.set(0, 2, 3.0f);
    matmul_a.set(1, 0, 4.0f);
    matmul_a.set(1, 1, 5.0f);
    matmul_a.set(1, 2, 6.0f);

    matmul_b.set(0, 0, 7.0f);
    matmul_b.set(0, 1, 8.0f);
    matmul_b.set(1, 0, 9.0f);
    matmul_b.set(1, 1, 10.0f);
    matmul_b.set(2, 0, 11.0f);
    matmul_b.set(2, 1, 12.0f);

    Tensor matmul_c = matmul_a.matmul(matmul_b);

    assert(matmul_c.shape()[0] == 2);
    assert(matmul_c.shape()[1] == 2);

    assert(matmul_c.at(0, 0) == 58.0f);
    assert(matmul_c.at(0, 1) == 64.0f);
    assert(matmul_c.at(1, 0) == 139.0f);
    assert(matmul_c.at(1, 1) == 154.0f);
    
    // autograd   
    Tensor weight({2, 2});

    weight.set_requires_grad(true);

    weight.set_grad(0, 0.5f);
    weight.set_grad(1, -1.0f);


    return 0;
}
