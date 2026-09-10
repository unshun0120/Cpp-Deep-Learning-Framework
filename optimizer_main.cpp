#include <cassert>

#include "optimizer.hpp"
#include "tensor.hpp"

int main() {
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

    return 0;
}
