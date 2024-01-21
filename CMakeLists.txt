cmake_minimum_required(VERSION 3.5)
project(torch_app)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(torch_app torch_app.cpp)
target_compile_features(torch_app PUBLIC cxx_range_for)
target_link_libraries(torch_app ${TORCH_LIBRARIES})

add_executable(Torch_images Torch_images.cpp)
target_compile_features(Torch_images PUBLIC cxx_range_for)
target_link_libraries(Torch_images ${TORCH_LIBRARIES})
target_link_libraries(Torch_images "${OpenCV_LIBS}")

add_executable(Torch_diamonds Torch_diamonds.cpp)
target_compile_features(Torch_diamonds PUBLIC cxx_range_for)
target_link_libraries(Torch_diamonds ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Torch_cpp>)
endif (MSVC)