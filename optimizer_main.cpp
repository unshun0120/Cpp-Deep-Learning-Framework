#include <cassert>

#include "optimizer.hpp"
#include "tensor.hpp"

int main() {
    // sgd step
    Tensor parameter({2, 2});
    Tensor gradient({2, 2});

    parameter.set(0, 1.0f);
    parameter.set(1, 2.0f);
    parameter.set(2, 3.0f);
    parameter.set(3, 4.0f);

    gradient.set(0, 0.5f);
    gradient.set(1, -1.0f);
    gradient.set(2, 2.0f);
    gradient.set(3, -0.5f);

    sgd_step(parameter, gradient, 0.1f);

    assert(parameter.at(0) == 0.95f);
    assert(parameter.at(1) == 2.1f);
    assert(parameter.at(2) == 2.8f);
    assert(parameter.at(3) == 4.05f);
    
    Tensor bias({3});
    Tensor bias_grad({3});

    bias.set(0, 1.0f);
    bias.set(1, 2.0f);
    bias.set(2, 3.0f);

    bias_grad.set(0, 0.5f);
    bias_grad.set(1, -1.0f);
    bias_grad.set(2, 2.0f);

    sgd_step(bias, bias_grad, 0.1f);

    assert(bias.at(0) == 0.95f);
    assert(bias.at(1) == 2.1f);
    assert(bias.at(2) == 2.8f);

    return 0;
}
