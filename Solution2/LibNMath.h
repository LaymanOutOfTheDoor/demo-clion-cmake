//
// Created by 56585 on 2023/7/19.
//

#ifndef INTERVIEW_LIBNMATH_H
#define INTERVIEW_LIBNMATH_H

#include <iostream>

template<typename T, typename E> T Mul(T a, E b) {
    return a * (T)b;
}

template<>
std::string Mul<std::string, int>(std::string a, int b);


#endif //INTERVIEW_LIBNMATH_H
