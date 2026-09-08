#pragma once

#include "tensor.hpp"

Tensor relu(const Tensor& input);

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

float mse_loss(const Tensor& prediction, const Tensor& target);

Tensor mse_loss_backward(const Tensor& prediction, const Tensor& target);

Tensor relu_backward(const Tensor& input, const Tensor& grad_output);




