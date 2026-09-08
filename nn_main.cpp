#include <cassert>

#include "nn.hpp"
#include "tensor.hpp"

int main() {

    // Relu
    Tensor x({2, 3});

    x.set(0, -2.0f);
    x.set(1, -0.5f);
    x.set(2, 0.0f);
    x.set(3, 1.5f);
    x.set(4, 3.0f);
    x.set(5, -4.0f);

    Tensor y = relu(x);

    assert(y.shape() == x.shape());
    assert(y.numel() == x.numel());

    assert(y.at(0) == 0.0f);
    assert(y.at(1) == 0.0f);
    assert(y.at(2) == 0.0f);
    assert(y.at(3) == 1.5f);
    assert(y.at(4) == 3.0f);
    assert(y.at(5) == 0.0f);

    assert(x.at(0) == -2.0f);
    assert(x.at(1) == -0.5f);
    assert(x.at(3) == 1.5f);
    assert(x.at(4) == 3.0f);
    
    // Linear
    Tensor input({2, 2});
    Tensor weight({2, 3});
    Tensor bias({3});

    input.set(0, 0, 1.0f);
    input.set(0, 1, 2.0f);
    input.set(1, 0, 3.0f);
    input.set(1, 1, 4.0f);

    weight.set(0, 0, 1.0f);
    weight.set(0, 1, 2.0f);
    weight.set(0, 2, 3.0f);
    weight.set(1, 0, 4.0f);
    weight.set(1, 1, 5.0f);
    weight.set(1, 2, 6.0f);

    bias.set(0, 1.0f);
    bias.set(1, 2.0f);
    bias.set(2, 3.0f);

    Tensor output = linear(input, weight, bias);

    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 3);

    assert(output.at(0, 0) == 10.0f);
    assert(output.at(0, 1) == 14.0f);
    assert(output.at(0, 2) == 18.0f);

    assert(output.at(1, 0) == 20.0f);
    assert(output.at(1, 1) == 28.0f);
    assert(output.at(1, 2) == 36.0f);

    
    // MSE Loss
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

    float loss = mse_loss(prediction, target);

    assert(loss == 1.25f);


    return 0;
}
