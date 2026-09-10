
#include <cassert>

#include "tensor.hpp"

using namespace std;

void sgd_step(Tensor& parameter, const Tensor& gradient, float learning_rate) {
    assert(parameter.shape() == gradient.shape());

    for(int i = 0; i < parameter.shape()[0]; i++) {
        for(int j = 0; j < parameter.shape()[1]; j++) {
            parameter.set(i, j, parameter.at(i, j) - learning_rate * gradient.at(i, j));
        }
    }
}








