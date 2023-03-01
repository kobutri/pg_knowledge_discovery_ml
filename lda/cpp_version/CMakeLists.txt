cmake_minimum_required(VERSION 3.25)
project(cpp_version)

set(CMAKE_CXX_STANDARD 20)

add_executable(cpp_version main.cpp)

find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(cpp_version PRIVATE JsonCpp::JsonCpp)
