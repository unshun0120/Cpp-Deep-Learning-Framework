#include<cassert>
#include<iostream>

#include "tensor.hpp"

using namespace std;

int main() {    
    Tensor a({2, 3});
    Tensor b({2, 3});

    for(size_t i = 0; i < 6; i++) {
        a.set(i, i+1);
        b.set(i, 10+i*10);
    }

    for(size_t i = 0; i < 6; i++){
        cout << "a" << a.at(i) << endl;
        cout << "b" << b.at(i) << endl;
    }
    
    Tensor c = a.add(b);

    assert(c.numel() == 6);
    assert(c.shape() == a.shape());

    assert(c.at(0) == 11.0f);
    assert(c.at(1) == 22.0f);
    assert(c.at(5) == 66.0f);

    assert(a.at(0) == 1.0f);
    assert(b.at(0) == 10.0f); 

    return 0;
}
