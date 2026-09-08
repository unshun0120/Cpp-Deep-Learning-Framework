#include<cassert>

#include "tensor.hpp"
#include "nn.hpp"

// Relu
Tensor relu(const Tensor& input) {
    Tensor result(input.shape());

    for(size_t i = 0; i < input.numel(); i++) {
        if(input.at(i) < 0.0f)
            result.set(i, 0.0f);
        else
            result.set(i, input.at(i));
    }

    return result;
}

// Linear
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    assert(input.shape().size() == 2);
    assert(weight.shape().size() == 2);
    assert(bias.shape().size() == 1);

    assert(input.shape()[1] == weight.shape()[0]);
    assert(weight.shape()[1] == bias.shape()[0]);
    

    // output = input * W + bias
    Tensor output = input.matmul(weight);
    
    for(size_t i = 0; i < output.shape()[0]; i++) {
        for(size_t j = 0; j < output.shape()[1]; j++) {
            float sum = output.at(i, j) + bias.at(j);
            output.set(i, j, sum);
        }
    }
    
    return output;
}


// MSE Loss
float mse_loss(const Tensor& prediction, const Tensor& target) {
    assert(prediction.shape() == target.shape());
    
    float loss = 0.0f;

    for(size_t i = 0; i < prediction.numel(); i++) {
        float diff = prediction.at(i) - target.at(i);
        loss += diff * diff;
    }
    loss /= prediction.numel();
        
    return loss;
}









