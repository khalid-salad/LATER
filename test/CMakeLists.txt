include_directories(
        ../QR/include
        $CUTLASS_DIR/include/
        $CUTLASS_DIR/tools/util/include
)

configure_file(check_qr.jl check_qr.jl COPYONLY)

file(GLOB TEST_SOURCES "*.cu")

foreach(SRC ${TEST_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    message(${NAME})
    add_executable(${NAME} "${NAME}.cu")
    set_property(TARGET ${NAME}
            PROPERTY CUDA_SEPARABLE_COMPILATION) 
    set_property(TARGET ${NAME} PROPERTY CUDA_ARCHITECTURES native)
    target_link_libraries(${NAME} qr utility evd chol ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
endforeach()
