cmake_minimum_required(VERSION 3.10)

project(Monte_Carlo_Portfolio_Simulation VERSION 1.0)

include(FetchContent)

FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_SHALLOW TRUE
  GIT_TAG tags/v1.0.0
)

FetchContent_Declare(
  libcurl
  GIT_REPOSITORY https://github.com/curl/curl.git
  GIT_SHALLOW TRUE
  GIT_TAG curl-8_10_1  
)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_SHALLOW TRUE
  GIT_TAG 3.4
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUTODIFF_DEBUG")

FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(libcurl) 
FetchContent_MakeAvailable(simdjson)

set(PROJ portfolio_simulation)
set(INC_DIR ${CMAKE_SOURCE_DIR}/inc)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

file(MAKE_DIRECTORY ${BIN_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SRC_FILES "${SRC_DIR}/*.cpp")

add_executable(${PROJ} ${SRC_FILES})

target_include_directories(${PROJ} PRIVATE ${INC_DIR} ${libcurl_SOURCE_DIR}/include)
target_link_libraries(${PROJ} PRIVATE libcurl eigen)

set_target_properties(${PROJ} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})