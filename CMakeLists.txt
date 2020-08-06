cmake_minimum_required(VERSION 2.8.4)

project( VlFeatTest )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(IS_MSVC)
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

option(BUILD_EXAMPLES "Whether to enable CMAKE generation of example projects" OFF)
option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)

# Find packages

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

#find_package( VLFeat REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( OpenCV REQUIRED )

if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
endif()

if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()


###############################################################################
# 3rd #########################################################################
###############################################################################

add_subdirectory(third_party)

###############################################################################
# Lib #########################################################################
###############################################################################

include_directories (BEFORE "third_party")
include_directories (BEFORE "src")

file(GLOB_RECURSE HDRS "src/VlFeatExtraction/*.hpp") 
file(GLOB_RECURSE SRCS "src/VlFeatExtraction/*.cpp") 

add_library(VlFeatExtraction STATIC
	${HDRS} ${SRCS}
)
target_link_libraries(VlFeatExtraction
	Eigen3::Eigen 
	vlfeat
	${OpenCV_LIBS}
)
add_dependencies(VlFeatExtraction 
  vlfeat
)

###############################################################################
# Example projects ############################################################
###############################################################################
if(BUILD_EXAMPLES)
  message(STATUS "Enabling CMAKE generation of example projects")
  
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

  file(GLOB HDRS include/*.h include/*.hpp)
  file(GLOB SRCS src/*.cpp)

  ###############################################################################
  add_executable(Example_ExtractAndMatch_TwoView
    ${HDRS} src/main.cpp
  )
  target_link_libraries(Example_ExtractAndMatch_TwoView 
    Eigen3::Eigen 
    vlfeat
    ${OpenCV_LIBS}
    VlFeatExtraction
  )
  add_dependencies(Example_ExtractAndMatch_TwoView 
    VlFeatExtraction
  )
endif()

