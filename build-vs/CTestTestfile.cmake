# CMake generated Testfile for 
# Source directory: C:/Users/Brandon/CudaGame
# Build directory: C:/Users/Brandon/CudaGame/build-vs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[TestRunner]=] "C:/Users/Brandon/CudaGame/build-vs/bin/tests/Debug/TestRunner.exe")
  set_tests_properties([=[TestRunner]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Brandon/CudaGame/CMakeLists.txt;533;add_test;C:/Users/Brandon/CudaGame/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[TestRunner]=] "C:/Users/Brandon/CudaGame/build-vs/bin/tests/Release/TestRunner.exe")
  set_tests_properties([=[TestRunner]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Brandon/CudaGame/CMakeLists.txt;533;add_test;C:/Users/Brandon/CudaGame/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[TestRunner]=] "C:/Users/Brandon/CudaGame/build-vs/bin/tests/MinSizeRel/TestRunner.exe")
  set_tests_properties([=[TestRunner]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Brandon/CudaGame/CMakeLists.txt;533;add_test;C:/Users/Brandon/CudaGame/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[TestRunner]=] "C:/Users/Brandon/CudaGame/build-vs/bin/tests/RelWithDebInfo/TestRunner.exe")
  set_tests_properties([=[TestRunner]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Brandon/CudaGame/CMakeLists.txt;533;add_test;C:/Users/Brandon/CudaGame/CMakeLists.txt;0;")
else()
  add_test([=[TestRunner]=] NOT_AVAILABLE)
endif()
subdirs("_deps/glfw-build")
subdirs("_deps/glad-build")
subdirs("_deps/assimp-build")
subdirs("_deps/googletest-build")
