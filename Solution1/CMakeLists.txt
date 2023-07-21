# Solution1/CMakeLists.txt


set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SRC_LIST_1 ${SOURCE_DIR}/*.cpp)
# 添加可执行文件，并将对应的源文件添加进来
add_executable(Solution1 ${SRC_LIST_1})

# 其他配置和命令...