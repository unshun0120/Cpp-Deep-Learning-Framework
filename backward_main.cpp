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
    
    // grad_input of Linear backward

    Tensor weight({2, 3});
    Tensor grad_output_2({2, 3});

    weight.set(0, 0, 1.0f);
    weight.set(0, 1, 2.0f);
    weight.set(0, 2, 3.0f);
    weight.set(1, 0, 4.0f);
    weight.set(1, 1, 5.0f);
    weight.set(1, 2, 6.0f);

    grad_output_2.set(0, 0, 1.0f);
    grad_output_2.set(0, 1, 2.0f);
    grad_output_2.set(0, 2, 3.0f);
    grad_output_2.set(1, 0, 4.0f);
    grad_output_2.set(1, 1, 5.0f);
    grad_output_2.set(1, 2, 6.0f);

    Tensor grad_input_2 = linear_backward_input(grad_output_2, weight);

    assert(grad_input_2.shape()[0] == 2);
    assert(grad_input_2.shape()[1] == 2);

    assert(grad_input_2.at(0, 0) == 14.0f);
    assert(grad_input_2.at(0, 1) == 32.0f);
    assert(grad_input_2.at(1, 0) == 32.0f);
    assert(grad_input_2.at(1, 1) == 77.0f);
    
    // grad_weight of Linear backward
    Tensor input_2({2, 2});
    Tensor grad_output_3({2, 3});

    input_2.set(0, 0, 1.0f);
    input_2.set(0, 1, 2.0f);
    input_2.set(1, 0, 3.0f);
    input_2.set(1, 1, 4.0f);

    grad_output_3.set(0, 0, 1.0f);
    grad_output_3.set(0, 1, 2.0f);
    grad_output_3.set(0, 2, 3.0f);
    grad_output_3.set(1, 0, 4.0f);
    grad_output_3.set(1, 1, 5.0f);
    grad_output_3.set(1, 2, 6.0f);

    Tensor grad_weight = linear_backward_weight(input_2, grad_output_3);

    assert(grad_weight.shape()[0] == 2);
    assert(grad_weight.shape()[1] == 3);

    assert(grad_weight.at(0, 0) == 13.0f);
    assert(grad_weight.at(0, 1) == 17.0f);
    assert(grad_weight.at(0, 2) == 21.0f);

    assert(grad_weight.at(1, 0) == 18.0f);
    assert(grad_weight.at(1, 1) == 24.0f);
    assert(grad_weight.at(1, 2) == 30.0f);
    
    // grad_bias of Linear backward
    Tensor grad_output_4({2, 3});

    grad_output_4.set(0, 0, 1.0f);
    grad_output_4.set(0, 1, 2.0f);
    grad_output_4.set(0, 2, 3.0f);

    grad_output_4.set(1, 0, 4.0f);
    grad_output_4.set(1, 1, 5.0f);
    grad_output_4.set(1, 2, 6.0f);

    Tensor grad_bias = linear_backward_bias(grad_output_4);

    assert(grad_bias.shape().size() == 1);
    assert(grad_bias.shape()[0] == 3);

    assert(grad_bias.at(0) == 5.0f);
    assert(grad_bias.at(1) == 7.0f);
    assert(grad_bias.at(2) == 9.0f);


    return 0;
}
