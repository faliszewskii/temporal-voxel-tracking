cmake_minimum_required(VERSION 3.29)
project(frame_generator C)

set(CMAKE_C_STANDARD 23)

add_library(frame_generator SHARED library.c)
