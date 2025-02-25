cmake_minimum_required(VERSION 3.12)

project(DifferentiableOwen)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

option(WITH_GEOGRAM "Build with geogram" OFF)
option(WITH_UTK "Build with UTK" OFF)
option(OMP_TRANSPORT "Disable the use of omp for 2D OT computation" ON)

include(openmp)
include(cli11)

IF (${WITH_GEOGRAM})
    # include(geogram)
    # message("Adding with geogram flagg")
    # add_compile_definitions(WITH_GEOGRAM)
ENDIF()

if (${WITH_UTK})
    include(utk)
ENDIF()

IF (OMP_TRANSPORT)
    message("Enabling OMP for OT in 2D")
    add_compile_definitions(WITH_OMP)
ENDIF()

ADD_LIBRARY(SOURCES
    # Utils
    src/utils/Array.cpp
    src/utils/BinaryUtils.cpp
    src/utils/ProgressBar.cpp
    src/utils/Logger.cpp

    # Losses
    src/loss/fourier/PCF.cpp
    src/loss/semi_discrete_ot_2d/transport.cpp
    src/loss/semi_discrete_ot_3d/geogram_wrapper.cpp
    src/loss/gbn/gbn.cpp
    src/loss/integration/Integrands.cpp
    src/loss/loss.cpp

    # Owen
    src/SmoothOwen/SmoothOwen.cpp
    # src/SmoothOwen/SmoothPermutationOwen.cpp
)
add_compile_definitions(DATA_PATH="${CMAKE_SOURCE_DIR}/data")

IF (${WITH_GEOGRAM})
    target_link_libraries(SOURCES OpenMP::OpenMP_CXX)
    # target_link_libraries(SOURCES OpenMP::OpenMP_CXX geogram exploragram)
ELSE()
    target_link_libraries(SOURCES OpenMP::OpenMP_CXX)
ENDIF()

target_include_directories(SOURCES PRIVATE ${CMAKE_SOURCE_DIR}/src)

IF (${WITH_GEOGRAM})
    add_executable(OwenDiff3D src/main/main3d.cpp)
    target_include_directories(OwenDiff3D PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(OwenDiff3D SOURCES CLI11::CLI11)

    
    add_executable(W23D src/main/eval_3d.cpp)
    target_include_directories(W23D PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(W23D SOURCES CLI11::CLI11)
ENDIF()

function(create_exe exename mainname)
    add_executable(${exename} src/main/main_${mainname}.cpp)
    target_include_directories(${exename} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${exename} SOURCES CLI11::CLI11)
endfunction()

create_exe(DifferentiableOwenW2 w2)
create_exe(DifferentiableOwenGBN gbn)
create_exe(DifferentiableOwenInt int)
create_exe(DifferentiableOwenpcf pcf)

create_exe(DifferentiableOwenProgressiveW2 progressive_w2)
create_exe(DifferentiableOwenProgressiveGBN progressive_gbn)
create_exe(DifferentiableOwenProgressiveInt progressive_int)