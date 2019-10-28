cmake_minimum_required(VERSION 2.8)


project(example_project)

include_directories("/usr/local/include")

LINK_DIRECTORIES("/usr/local/lib")

LINK_LIBRARIES(boost_system boost_filesystem)

find_package(OpenCV REQUIRED)


add_executable(demhist main.cpp)

target_link_libraries(demhist ${OpenCV_LIBS})
