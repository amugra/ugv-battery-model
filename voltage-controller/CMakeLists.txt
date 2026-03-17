cmake_minimum_required(VERSION 3.13...3.27)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(discharge-experiment C CXX ASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(discharge-experiment
    main.c
    usb_descriptors.c
)
target_link_libraries(discharge-experiment
    pico_stdlib
    hardware_dma
    hardware_adc
    pico_multicore
    tinyusb_device
    tinyusb_board
)
target_include_directories(discharge-experiment
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)
pico_enable_stdio_usb(discharge-experiment 0)
pico_enable_stdio_uart(discharge-experiment 0)

pico_add_extra_outputs(discharge-experiment)            

add_custom_target(copy-compile-commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
)
