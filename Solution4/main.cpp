#include "LibNMath.h"

int main() {
    std::string str = "aabb|";
    auto data = Mul(str, 10);
    std::cout << data << std::endl;
    return 0;
}
