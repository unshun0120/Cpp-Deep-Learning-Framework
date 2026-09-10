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
// Sigma[(pred - target)^2] / N 
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


/*
 forward : input -> Linear -> Relu -> loss
 bakward : loss -> mse backward -> relu output -> relu backward -> linear output -> linear backward -> linear input
 =
 X -> [Linear] -> Z=XW+b -> [ReLU] -> A = ReLU(Z) -> [Loss] -> L
 
 backward : 收到L想求∂L/∂A -> 收到∂L/∂A想求∂L/∂Z -> 收到∂L/∂Z想求∂L/∂X
 backward就是在算""某個參數改變時, loss會受到多少影響"
 傳回前面的gradient = 後面傳來的gradient * 這一層自己的derivative
*/

// MSE loss backward gradient
// 求∂L/∂A, 對prediction微分
// 對MSE的predict微分, Sigma[(pred - target)^2] / N -> 2*(pred-target) / N
Tensor mse_loss_backward(const Tensor& prediction, const Tensor& target) {
    assert(prediction.shape() == target.shape());
    
    Tensor gradient(prediction.shape());
    for(size_t b = 0; b < prediction.shape()[0]; b++) {
        for(size_t i = 0; i < prediction.shape()[1]; i++) {
            gradient.set(b, i, 2.0f * (prediction.at(b, i) - target.at(b, i)) / prediction.numel());
        }
    }
    
    return gradient;
}

// 求∂L/∂Z = ∂L/∂A * ∂A/∂Z -> if Z > 0, ∂A/∂Z = 1, if Z <= 0, ∂A/∂Z = 0  
// if Z > 0 -> ∂L/∂Z = ∂L/∂A, if Z <= 0 -> ∂L/∂Z = 0
// 收到經過ReLU的loss回傳的gradient, 在計算ReLU前面input對loss的影響多大(grad_input) 
// 微分/gradient的意義: 某個變數改變時對loss的影響程度, loss會改變多少
// 為什麼training需要gradient: 模型要更新weight和bias, 就必須知道參數往哪個方向改會讓loss下降, optimizer才能正確更新參數
Tensor relu_backward(const Tensor& input, const Tensor& grad_output) {
    assert(input.shape() == grad_output.shape());
    
    Tensor grad_input(grad_output.shape());
    
    for(size_t b = 0; b < input.shape()[0]; b++) {
        for(size_t i = 0; i < input.shape()[1]; i++) { 
            if(input.at(b, i) > 0)
                grad_input.set(b, i, grad_output.at(b, i));
            else
                grad_input.set(b, i, 0.0f);
        }
    }

    return grad_input;
}

// Linear backward 的 grad_input
// XW+b的X改一點對loss的影響
// 求∂L/∂X = ∂L/∂Z * ∂Z/∂X, 因為Z = XW+b所以∂Z/∂X = W, 得∂L/∂X = ∂L/∂Z * W, 
// Z[b,j]=i∑X[b,i]*W[i,j]+bias[j], 某個X[b,0]會同時影響Z[b,0] Z[b,1] Z[b,2], 要知道X[b,0]對loss的影響就都要算進去
// 矩陣就是 grad X = grad Z * W^T, grad_input[b,i]=j∑grad_output[b,j]*W[i,j]
// grad_output: [batch, out_features]
// weight:      [in_features, out_features]
Tensor linear_backward_input(const Tensor& grad_output, const Tensor& weight) {
    assert(grad_output.shape().size() == 2);
    assert(weight.shape().size() == 2);
    assert(grad_output.shape()[1] == weight.shape()[1]);

    Tensor grad_input({grad_output.shape()[0], weight.shape()[0]});
    
    // batch
    for(size_t b = 0; b < grad_input.shape()[0]; b++) {
        // input feature
        for(size_t i = 0; i < weight.shape()[0]; i++) {
            float sum = 0;
            // output feature
            for(size_t j = 0; j < weight.shape()[1]; j++) {
                sum += grad_output.at(b, j) * weight.at(i, j);
            }
            grad_input.set(b, i, sum);
        }
    }
    
    return grad_input;
}

// Linear backward 的 grad_weight
// XW+b的某一個weight W[i,j]改變, 對loss的影響
// ∂L/∂W[i,j]是optimizer拿來更新weight的東西
// 同一個weight W[i,j]會被整個batch裡的資料共同使用, 所以這個weight對loss的影響要把整個batch的影響加起來
// grad_weight[i,j]=b∑X[b,i]*grad_output[b,j]
// input       = {batch, in_features}
// grad_output = {batch, out_features}
Tensor linear_backward_weight(const Tensor& input, const Tensor& grad_output) {
    assert(input.shape().size() == 2);
    assert(grad_output.shape().size() == 2);
    assert(input.shape()[0] == grad_output.shape()[0]);

    Tensor grad_weight({input.shape()[1], grad_output.shape()[1]});

    for(size_t i = 0; i < grad_weight.shape()[0]; i++) {
        for(size_t j = 0; j < grad_weight.shape()[1]; j++) {
            float sum = 0;
            for(size_t b = 0; b < input.shape()[0]; b++) {
                sum += input.at(b, i)*grad_output.at(b, j);
            }
            grad_weight.set(i, j, sum);
        }
    }
    
    return grad_weight;
}

// Linear backward 的 grad_bias
// XW+b的bias b改變, 對loss的影響
// 同一個bias b[i]會被所有batch使用
// ∂Z/∂bias = 1
// grad_bias[i]=b∑grad_output[b,i]
// grad_output = {batch, out_features}
Tensor linear_backward_bias(const Tensor& grad_output) {
    assert(grad_output.shape().size() == 2);
    
    Tensor grad_bias({grad_output.shape()[1]});

    for(size_t i = 0; i < grad_output.shape()[1]; i++) {
        float sum = 0;
        for(size_t b = 0; b < grad_output.shape()[0]; b++) {
            sum += grad_output.at(b, i);
        }
        grad_bias.set(i, sum);
    }
    
    return grad_bias;
}





















