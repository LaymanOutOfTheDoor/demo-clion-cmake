//
// Created by 56585 on 2023/7/19.
//

#ifndef INTERVIEW_LIBNMATH_H
#define INTERVIEW_LIBNMATH_H

#include <iostream>

template<typename T, typename E>
T Mul(T a, E b) {
    return a * (T)b;
}

template<>
inline std::string Mul<std::string, int> (const std::string a, int b) {
    std::string res{};
    for (int i=0; i<b; i++) {
        res += a;
    }
    return res;
}


#endif //INTERVIEW_LIBNMATH_H
