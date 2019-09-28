cmake_minimum_required(VERSION 3.2)

# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PWD})   # move to current folder from build folder
set(USE_AVX_INSTRUCTIONS TRUE)
if(NOT CMAKE_BUILD_RELEASE)
    set(CMAKE_BUILD_RELEASE FALSE)
endif()
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# project name
project(Annotation)

# library
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

foreach(loop_var gflags glog)
    link_directories(/usr/local/opt/${loop_var}/lib)
    include_directories(/usr/local/opt/${loop_var}/include)
    list(APPEND CELLAR_LIBS ${loop_var})
endforeach(loop_var)


if(NOT DEFINED HEADERS)
    file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/*.h*)
endif()

if(NOT DEFINED SOURCES)
    file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/*.cpp)
endif()

find_package(OpenCV)


macro(build_tools name)
   add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/src/${name}.cpp ${HEADERS} ${SOURCES})
   target_link_libraries(${name}
       ${OpenCV_LIBS}                  
        caffe
        tesseract
        ${CELLAR_LIBS}
   )
endmacro()


build_tools(annotateRect)
build_tools(extractRect)

