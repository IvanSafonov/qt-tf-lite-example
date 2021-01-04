cmake_minimum_required(VERSION 3.16.0)
project(qt-tf-lite-example)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Qt5 COMPONENTS Gui REQUIRED)

add_executable(qt-tf-lite-example
  main.cpp
  model.cpp
)

add_subdirectory(
  "/opt/tensorflow/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

target_link_libraries(qt-tf-lite-example tensorflow-lite ${CMAKE_DL_LIBS} Qt5::Gui)
target_include_directories(qt-tf-lite-example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
