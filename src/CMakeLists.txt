cmake_minimum_required( VERSION 2.6 )
set ( CMAKE_BUILD_TYPE Debug )
add_definitions ( -Wall )

find_package(OpenCV REQUIRED)

set ( SVMTest_HDRS
        SvmLightLib.h
        svm/kernel.h
        svm/svm_struct_learn.h
        svm/svm_struct_common.h
        svm/svm_struct_api.h
        svm/svm_struct_api_types.h
        svm/svm_learn.h
        svm/svm_common.h
)

set ( SVMTest_SRCS
	main.cpp
        train_HOG.cpp
	SvmLightLib.cpp
        svm/svm_struct_main.c
        svm/svm_struct_learn.c
        svm/svm_struct_common.c
        svm/svm_struct_classify.c
        svm/svm_struct_api.c
        svm/svm_learn.c
        svm/svm_learn_main.c
        svm/svm_hideo.c
        svm/svm_common.c
        svm/svm_classify.c
)

add_executable ( SVMTest ${SVMTest_SRCS} )
target_link_libraries ( SVMTest ${OpenCV_LIBS} )
