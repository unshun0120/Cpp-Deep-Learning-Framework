#include <cassert>

#include "tensor.hpp"

int main() {
    Tensor t({2, 3});

    for(size_t i = 0; i < t.numel(); i++) {
        t.set(i, static_cast<float>(i));
    }
    
    t.reshape({3, 2});

    assert(t.shape()[0] == 3);
    assert(t.shape()[1] == 2);  

    assert(t.numel() == 6);

    assert(t.at(0) == 0.0f);
    assert(t.at(1) == 1.0f);
    assert(t.at(2) == 2.0f);
    assert(t.at(3) == 3.0f);
    assert(t.at(4) == 4.0f);
    assert(t.at(5) == 5.0f);
    
    t.reshape({2, 4});

    return 0;
}
