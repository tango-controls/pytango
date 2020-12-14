cmake_minimum_required(VERSION 3.0.2)
set(CMP0048 NEW)

project(_tango LANGUAGES CXX)

# Set PROJECT_VERSION and PACKAGE_VERSION from git tags
include(cmake/project_version.cmake)

if(NOT WIN32)
  message(FATAL_ERROR "This file is only targeting Windows builds - aborting!")
endif(NOT WIN32)

add_definitions(-DWIN32)
add_definitions(-DNDEBUG)
add_definitions(-D_WINSTATIC)
add_definitions(-DNPY_NO_DEPRECATED_API)
add_definitions(-DBOOST_PYTHON_STATIC_LIB)
add_definitions(-DBOOST_ALL_NO_LIB)
add_definitions(-DZMQ_STATIC)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(PY_TARGET "64")
  add_definitions(-D_64BITS)
  if(MSVC14)
    add_definitions(-D_TIMERS_T_)
  endif(MSVC14)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PY_TARGET "32")
  add_definitions(-DJPG_USE_ASM)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Unknown architecture - aborting!")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

set(SOURCES
        ext/api_util.cpp
        ext/archive_event_info.cpp
        ext/server/attr.cpp
        ext/attr_conf_event_data.cpp
        ext/server/attribute.cpp
        ext/server/auto_monitor.cpp
        ext/attribute_alarm_info.cpp
        ext/attribute_dimension.cpp
        ext/attribute_event_info.cpp
        ext/attribute_info.cpp
        ext/attribute_info_ex.cpp
        ext/attribute_proxy.cpp
        ext/base_types.cpp
        ext/callback.cpp
        ext/change_event_info.cpp
        ext/server/command.cpp
        ext/command_info.cpp
        ext/connection.cpp
        ext/constants.cpp
        ext/data_ready_event_data.cpp
        ext/database.cpp
        ext/db.cpp
        ext/dev_command_info.cpp
        ext/dev_error.cpp
        ext/device_attribute.cpp
        ext/device_attribute_config.cpp
        ext/device_attribute_history.cpp
        ext/server/device_class.cpp
        ext/device_data.cpp
        ext/device_data_history.cpp
        ext/server/device_impl.cpp
        ext/device_info.cpp
        ext/device_pipe.cpp
        ext/device_proxy.cpp
        ext/devintr_change_event_data.cpp
        ext/server/dserver.cpp
        ext/server/encoded_attribute.cpp
        ext/enums.cpp
        ext/event_data.cpp
        ext/exception.cpp
        ext/from_py.cpp
        ext/server/fwdAttr.cpp
        ext/group.cpp
        ext/group_reply.cpp
        ext/group_reply_list.cpp
        ext/locker_info.cpp
        ext/locking_thread.cpp
        ext/server/log4tango.cpp
        ext/server/multi_attribute.cpp
        ext/server/multi_class_attribute.cpp
        ext/periodic_event_info.cpp
        ext/server/pipe.cpp
        ext/pipe_event_data.cpp
        ext/pipe_info.cpp
        ext/poll_device.cpp
        ext/precompiled_header.cpp
        ext/pytango.cpp
        ext/pytgutils.cpp
        ext/pyutils.cpp
        ext/server/subdev.cpp
        ext/server/tango_util.cpp
        ext/time_val.cpp
        ext/to_py.cpp
        ext/server/user_default_attr_prop.cpp
        ext/server/user_default_pipe_prop.cpp
        ext/version.cpp
        ext/server/wattribute.cpp)

set(HEADERS
        ext/precompiled_header.hpp
        ext/base_types_numpy.hpp
        ext/fast_from_py_numpy.hpp
        ext/server/wattribute_numpy.hpp
        ext/device_attribute_numpy.hpp
        ext/to_py_numpy.hpp
        ext/server/attr.h
        ext/server/attribute.h
        ext/callback.h
        ext/server/command.h
        ext/defs.h
        ext/device_attribute.h
        ext/server/device_class.h
        ext/server/device_impl.h
        ext/device_pipe.h
        ext/exception.h
        ext/fast_from_py.h
        ext/from_py.h
        ext/server/pipe.h
        ext/pytgutils.h
        ext/pyutils.h
        ext/tango_numpy.h
        ext/tgutils.h
        ext/to_py.h)

set(${PROJECT_NAME} INCLUDE_DIRS "$ENV{TANGO_ROOT}/include")

if(MSVC90)
  # v9 used for Python 2
  set(VCSTR "90")
  set(BOOST_VERSION_ "1_70")
elseif(MSVC14)
  # v14 used for Python 3
  set(VCSTR "140")
  set(BOOST_VERSION_ "1_73")
endif()
set(ZMQ_VERSION_ "4_0_5")

set(RELEASE "Release_$ENV{PYTHON_VER}_${PY_TARGET}")
include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext")
include_directories(${PROJECT_NAME} PUBLIC "$ENV{BOOST_ROOT}/include/boost-${BOOST_VERSION_}")
include_directories(${PROJECT_NAME} PUBLIC "$ENV{PYTHON_ROOT}/include")
include_directories(${PROJECT_NAME} PUBLIC "$ENV{TANGO_ROOT}/include")
include_directories(${PROJECT_NAME} PUBLIC "$ENV{PYTHON_ROOT}/Lib/site-packages/numpy/core/include")
link_directories("$ENV{TANGO_ROOT}/bin")
link_directories("$ENV{TANGO_ROOT}/lib")
link_directories("$ENV{PYTHON_ROOT}/libs")
link_directories("$ENV{BOOST_ROOT}/lib")
message("TANGO at: $ENV{TANGO_ROOT}")
message("BOOST at: $ENV{BOOST_ROOT}")
message("PYTHON at: $ENV{PYTHON_ROOT}")
message("Release at: ${CMAKE_CURRENT_SOURCE_DIR}/${RELEASE}")
add_compile_options(/MT)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_compile_options(${PROJECT_NAME} PUBLIC "/MT")
set_target_properties( ${PROJECT_NAME}
  PROPERTIES
  SUFFIX ".pyd" )
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  COMPILE_OPTIONS "MT" )
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/${RELEASE}/tango"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/${RELEASE}/tango"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/${RELEASE}/tango")

if(MSVC90 AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
  # resolve VC 9 64bits complains
  set_source_files_properties(ext/base_types.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND "$ENV{PYTHON_ROOT}/python.exe"  ARGS "${CMAKE_CURRENT_SOURCE_DIR}/winsetup.py" ${RELEASE} dist temp_dist $(ConfigurationName) $(PlatformName)
  COMMENT "Building installer")


install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS "/bigobj")
set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS "/MT")
set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS "/LTCG /force:multiple /DYNAMICBASE:NO /OPT:REF /OPT:ICF /SUBSYSTEM:WINDOWS")
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_ROOT}/lib/libtango.lib
                                             $ENV{TANGO_ROOT}/lib/omniORB4.lib
                                             $ENV{TANGO_ROOT}/lib/omniDynamic4.lib
                                             $ENV{TANGO_ROOT}/lib/COS4.lib
                                             $ENV{TANGO_ROOT}/lib/omnithread.lib
                                             $ENV{TANGO_ROOT}/lib/msvcstub.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ws2_32 mswsock advapi32 comctl32 odbc32)
target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_ROOT}/lib/pthreadVC2.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_ROOT}/lib/libzmq-v${VCSTR}-mt-s-${ZMQ_VERSION_}.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_ROOT}/lib/libboost_python$ENV{PYTHON_VER}-vc${VCSTR}-mt-s-x${PY_TARGET}-${BOOST_VERSION_}.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{PYTHON_ROOT}/libs/python$ENV{PYTHON_VER}.lib)
