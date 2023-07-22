#cmake_minimum_required(VERSION 3.25)
#project(Interview)
#
#set(CMAKE_CXX_STANDARD 17)
#
#
#set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#file(GLOB SRC_LIST_0 ${SOURCE_DIR}/problem/*.cpp)
#file(GLOB SRC_LIST_1 ${SOURCE_DIR}/Solution1/*.cpp)
#file(GLOB SRC_LIST_2 ${SOURCE_DIR}/Solution2/*.cpp)
#file(GLOB SRC_LIST_3 ${SOURCE_DIR}/Solution3/*.cpp)
#
## 问题
#add_executable(Interview ${SRC_LIST_0} Solution2/main.cpp)
#
## 解决方案一
#add_executable(Solution1 ${SRC_LIST_1} Solution2/main.cpp)
## 解决方案二
#add_executable(Solution2 ${SRC_LIST_2} Solution2/main.cpp)
## 解决方案三
#add_executable(Solution3 ${SRC_LIST_3} Solution2/main.cpp)
#
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

cmake_minimum_required(VERSION 3.25)
project(Interview)
set(CMAKE_CXX_STANDARD 17)
set(HOME ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${HOME}/exe)

add_subdirectory(problem)

# 添加子目录Solution1
add_subdirectory(Solution1)

# 添加子目录Solution2
add_subdirectory(Solution2)

# 添加子目录Solution3
add_subdirectory(Solution3)

# 添加子目录Solution4
add_subdirectory(Solution4)
