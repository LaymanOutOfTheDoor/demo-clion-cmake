//
// Created by 56585 on 2023/7/19.
//

#include "LibNMath.h"
template<typename T, typename E>
T Mul(T a, E b) {
    return a * static_cast<T>(b);
}
template<>
std::string Mul<std::string, int> (std::string a, int b) {
    std::string res{};
    for (int i=0; i<b; i++) {
        res += a;
    }
    return res;
}