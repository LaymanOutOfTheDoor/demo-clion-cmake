//
// Created by 56585 on 2023/7/19.
//

#include "LibNMath.h"
template<>
std::string Mul<std::string, int>(std::string a, int b) {
    std::string res{};
    for (int i = 0; i < b; i++) {
        res += a;
    }
    return res;
}