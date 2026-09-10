#include<iostream>

#include "tensor.hpp"
#include "nn.hpp"
#include "optimizer.hpp"

using namespace std;

int main() {
    // initialize
    Tensor input({2, 1}), weight({1, 1}), bias({1}), target({2, 1});
    input.set(0, 1); input.set(1, 2);
    weight.set(0, 1); 
    bias.set(0, 0);
    target.set(0, 3);   target.set(1, 5);

    Tensor prediction = linear(input, weight, bias);
    float loss = mse_loss(prediction, target);
    cout << "fisr loss" << loss << endl;
    Tensor grad_output = mse_loss_backward(prediction, target);
    // Tensor backward_relu = relu_backward(input, grad_output);
    // Tensor backward_input = linear_backward_input(grad_output, backward_relu);
    Tensor backward_weight = linear_backward_weight(input, grad_output);
    Tensor backward_bias = linear_backward_bias(grad_output);        
    
    float learning_rate = 0.1;
    sgd_step(weight, backward_weight, learning_rate);
    sgd_step(bias, backward_bias, learning_rate);
    
    prediction = linear(input, weight, bias);
    loss = mse_loss(prediction, target);
    cout << "second loss" << loss << endl;

    return 0;
}













