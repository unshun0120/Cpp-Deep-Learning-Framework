#include <cassert>

#include "tensor.hpp"

using namespace std;

int main() {

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
}

