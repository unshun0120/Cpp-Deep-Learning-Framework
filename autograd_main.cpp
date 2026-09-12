#include <cassert>
#include <iostream>

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

    // topology
    Tensor a_4({2});
    Tensor b_4({2});
    Tensor e_4({2});

    Tensor c_4 = a_4.add(b_4);
    Tensor d_4 = c_4.add(e_4);
    
    std::vector<Tensor*> order = d_4.build_topology();
    assert(order.size() == 5);

    assert(order[0] == &a_4);
    assert(order[1] == &b_4);
    assert(order[2] == &c_4);
    assert(order[3] == &e_4);
    assert(order[4] == &d_4);

    // backward
    /* Tensor a_5({2});
    Tensor b_5({2});
    Tensor e_5({2});

    a_5.set_requires_grad(true);
    b_5.set_requires_grad(true);
    e_5.set_requires_grad(true);

    Tensor c_5 = a_5.add(b_5);
    Tensor d_5 = c_5.add(e_5);

    d_5.set_grad(0, 2.0f);
    d_5.set_grad(1, 3.0f);

    d_5.backward();

    assert(c_5.grad(0) == 2.0f);
    assert(c_5.grad(1) == 3.0f);

    assert(a_5.grad(0) == 2.0f);
    assert(a_5.grad(1) == 3.0f);

    assert(b_5.grad(0) == 2.0f);
    assert(b_5.grad(1) == 3.0f);

    assert(e_5.grad(0) == 2.0f);
    assert(e_5.grad(1) == 3.0f);*/ 

    // loss.backward
    Tensor a_6({1});
    Tensor b_6({1});

    a_6.set_requires_grad(true);
    b_6.set_requires_grad(true);

    a_6.set(0, 2.0f);
    b_6.set(0, 3.0f);

    Tensor c_6 = a_6.add(b_6);

    c_6.backward();

    assert(c_6.grad(0) == 1.0f);
    assert(a_6.grad(0) == 1.0f);
    assert(b_6.grad(0) == 1.0f);
    
    // backward multiply
    Tensor a_7({1});
    Tensor b_7({1});

    a_7.set_requires_grad(true);
    b_7.set_requires_grad(true);

    a_7.set(0, 2.0f);
    b_7.set(0, 3.0f);

    Tensor c_7 = a_7.multiply(b_7);

    c_7.backward();
    
    // backward add + multiply
    Tensor a_8({1});
    Tensor b_8({1});

    a_8.set_requires_grad(true);
    b_8.set_requires_grad(true);

    a_8.set(0, 2.0f);
    b_8.set(0, 3.0f);

    Tensor x_8 = a_8.multiply(b_8);
    Tensor y_8 = x_8.add(a_8);

    y_8.backward();

    cout << "a grad = " << a_8.grad(0) << endl;
    cout << "b grad = " << b_8.grad(0) << endl;

}

