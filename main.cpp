#include <cassert>

#include "tensor.hpp"

int main() {
    Tensor t({2, 3});

    t.fill(1.0f);
    t.set(2, 10.0f);

    for(int i = 0; i < t.numel(); i++) {
        if(i == 2)
            assert(t.at(2) == 10.0f);
        else
            assert(t.at(i) == 1.0f);
    }
    
    return 0;
}
