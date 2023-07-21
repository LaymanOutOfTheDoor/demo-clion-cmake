#include "LibNMath.h"
template std::string Mul<std::string, int>(std::string a, int b);
int main() {
    std::string str = "aabb|";
    auto data = Mul(str, 10);
    std::cout << data << std::endl;
    return 0;
}
