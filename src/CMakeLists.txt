CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(SOURCE_FILES main_training.cpp rts_forest.cpp rts_tree.cpp)
add_executable(rf_exe ${SOURCE_FILES})
add_custom_command(TARGET rf_exe
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rf_exe> ${CMAKE_CURRENT_BINARY_DIR})