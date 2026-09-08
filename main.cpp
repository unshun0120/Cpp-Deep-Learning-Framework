#include <cassert>

#include "tensor.hpp"

int main() {
    Tensor a({1, 2});
    Tensor b({2, 1});

    a.set(0, 0, 0.5f);
    a.set(0, 1, 1.0f);

    b.set(0, 0, 2.0f);
    b.set(1, 0, 0.5f);

    Tensor c = a.matmul(b);

    assert(c.shape()[0] == 1);
    assert(c.shape()[1] == 1);

    assert(c.at(0, 0) == 1.5f);

    return 0;
}
