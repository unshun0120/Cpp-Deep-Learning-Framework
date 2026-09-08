#include <cassert>

#include "nn.hpp"

int main() {
    // loss backward gradient
    Tensor prediction({2, 2});
    Tensor target({2, 2});

    prediction.set(0, 1.0f);
    prediction.set(1, 2.0f);
    prediction.set(2, 3.0f);
    prediction.set(3, 4.0f);

    target.set(0, 1.0f);
    target.set(1, 4.0f);
    target.set(2, 2.0f);
    target.set(3, 4.0f);

    Tensor grad = mse_loss_backward(prediction, target);

    assert(grad.shape() == prediction.shape());
    assert(grad.numel() == prediction.numel());

    assert(grad.at(0) == 0.0f);
    assert(grad.at(1) == -1.0f);
    assert(grad.at(2) == 0.5f);
    assert(grad.at(3) == 0.0f);

    // relu backward
    Tensor input({2, 3});
    Tensor grad_output({2, 3});

    input.set(0, -2.0f);
    input.set(1, -0.5f);
    input.set(2, 0.0f);
    input.set(3, 1.5f);
    input.set(4, 3.0f);
    input.set(5, -4.0f);

    grad_output.set(0, 1.0f);
    grad_output.set(1, 2.0f);
    grad_output.set(2, 3.0f);
    grad_output.set(3, 4.0f);
    grad_output.set(4, -5.0f);
    grad_output.set(5, 6.0f);

    Tensor grad_input = relu_backward(input, grad_output);

    assert(grad_input.shape() == input.shape());

    assert(grad_input.at(0) == 0.0f);
    assert(grad_input.at(1) == 0.0f);
    assert(grad_input.at(2) == 0.0f);
    assert(grad_input.at(3) == 4.0f);
    assert(grad_input.at(4) == -5.0f);
    assert(grad_input.at(5) == 0.0f);

    return 0;
}
