#include <cassert>
#include "tensor.hpp"

using namespace std;

int main() {
    // requires grad / grad
    Tensor weight({2, 2});

    assert(weight.requires_grad() == false);

    weight.set_requires_grad(true);
    assert(weight.requires_grad() == true);

    assert(weight.grad(0) == 0.0f);
    assert(weight.grad(1) == 0.0f);

    weight.set_grad(0, 0.5f);
    weight.set_grad(1, -1.0f);

    assert(weight.grad(0) == 0.5f);
    assert(weight.grad(1) == -1.0f);
    
    // operation add auto grad
    Tensor a({2});
    Tensor b({2});

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    a.set(0, 1.0f);
    a.set(1, 2.0f);

    b.set(0, 3.0f);
    b.set(1, 4.0f);

    Tensor c = a.add(b);

    assert(c.at(0) == 4.0f);
    assert(c.at(1) == 6.0f);

    assert(c.requires_grad() == true);

    assert(c.op() == OpType::Add);

    assert(c.num_parents() == 2);

    assert(c.parent(0) == &a);
    assert(c.parent(1) == &b);
    
    // backward
    Tensor a_2({2});
    Tensor b_2({2});

    a_2.set_requires_grad(true);
    b_2.set_requires_grad(true);

    Tensor c_2 = a_2.add(b_2);

    a_2.set_grad(0, 10.0f);
    a_2.set_grad(1, 20.0f);

    c_2.set_grad(0, 2.0f);
    c_2.set_grad(1, 3.0f);

    c_2.backward_add();

    assert(a_2.grad(0) == 12.0f);
    assert(a_2.grad(1) == 23.0f);

    assert(b_2.grad(0) == 2.0f);
    assert(b_2.grad(1) == 3.0f);
    
    // backward_2
    Tensor a_3({2});
    Tensor b_3({2});
    Tensor e_3({2});

    a_3.set_requires_grad(true);
    b_3.set_requires_grad(true);
    e_3.set_requires_grad(true);

    Tensor c_3 = a_3.add(b_3);
    Tensor d_3 = c_3.add(e_3);

    d_3.set_grad(0, 2.0f);
    d_3.set_grad(1, 3.0f);

    d_3.backward_add();
    c_3.backward_add();

    assert(c_3.grad(0) == 2.0f);
    assert(c_3.grad(1) == 3.0f);

    assert(a_3.grad(0) == 2.0f);
    assert(a_3.grad(1) == 3.0f);

    assert(b_3.grad(0) == 2.0f);
    assert(b_3.grad(1) == 3.0f);

    assert(e_3.grad(0) == 2.0f);
    assert(e_3.grad(1) == 3.0f);

}

