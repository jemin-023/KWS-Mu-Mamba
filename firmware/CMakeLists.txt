cmake_minimum_required(VERSION 3.16)

# Include engine directory as an extra component
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../engine")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kws_mu_mamba)
