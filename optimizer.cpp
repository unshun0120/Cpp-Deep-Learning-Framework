
#include <cassert>

#include "tensor.hpp"

using namespace std;

void sgd_step(Tensor& parameter, const Tensor& gradient, float learning_rate) {
    assert(parameter.shape() == gradient.shape());

    for(size_t i = 0; i < parameter.numel(); i++) {
        parameter.set(i, parameter.at(i) - learning_rate * gradient.at(i));
    }
}








