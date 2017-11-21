cmake_minimum_required(VERSION 3.0.2)
set(CMP0048 NEW)

project(_tango LANGUAGES CXX
        VERSION 1.0)
        
#need to define the version of the library
set(MAJOR_VERSION "9")
set(MINOR_VERSION "2")
set(PATCH_VERSION "2")

#convenient versions
set(LIBRARY_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
if(NOT WIN32)
include(FindPkgConfig)

pkg_search_module(TANGO_PKG REQUIRED tango)
endif(NOT WIN32)

if(WIN32)
add_definitions(-DWIN32)
#add_definitions(-D_WINDLL)
add_definitions(-DNDEBUG)
add_definitions(-D_WINSTATIC)
#add_definitions(-DTANGO_HAS_DLL)
#add_definitions(-DLOG4TANGO_HAS_DLL)
add_definitions(-D_WINSOCKAPI_)
add_definitions(-DNPY_NO_DEPRECATED_API)
add_definitions(-DBOOST_PYTHON_STATIC_LIB)
add_definitions(-DBOOST_ALL_NO_LIB)
add_definitions(-DZMQ_STATIC)
add_compile_options(-Zm200 /MP8 /bigobj /MT)
if(CMAKE_CL_64)
add_definitions(-D_64BITS)
if(MSVC14)
add_definitions(-D_TIMERS_T_)
endif(MSVC14)
else(CMAKE_CL_64)
add_definitions(-DJPG_USE_ASM)
endif(CMAKE_CL_64)
endif(WIN32)
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

if(WIN32)

  set(${PROJECT_NAME} INCLUDE_DIRS "$ENV{TANGO_PATH}/include")
  include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext")
  include_directories(${PROJECT_NAME} PUBLIC "$ENV{BOOST_ROOT}")
  include_directories(${PROJECT_NAME} PUBLIC "$ENV{PYTHONPATH}/include")
  include_directories(${PROJECT_NAME} PUBLIC "$ENV{TANGO_PATH}/include")
  include_directories(${PROJECT_NAME} PUBLIC "$ENV{PYTHONPATH}/Lib/site-packages/numpy/core/include")
  link_directories("$ENV{TANGO_PATH}/bin")
  link_directories("$ENV{TANGO_PATH}/lib")
  link_directories("$ENV{PYTHONPATH}/libs")
  message("$ENV{TANGO_PATH}/include")
  link_directories("$ENV{BOOST_LIBS}")
  message("$ENV{BOOST_LIBS}")
  add_compile_options(/MT)
  add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS}) 
  set_target_properties( ${PROJECT_NAME}
         PROPERTIES
         SUFFIX ".pyd" )
    set_target_properties( ${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Release/tango"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Release/tango"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Release/tango"
)

 add_custom_command(TARGET ${PROJECT_NAME}
           POST_BUILD
           COMMAND $ENV{PYTHONPATH}/python.exe  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/winsetup.py Release dist temp_dist $(ConfigurationName) $(PlatformName)
           COMMENT "Building msi and exe")
         

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

  set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS "/bigobj /MT")
  set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS "/LTCG /force:multiple /DYNAMICBASE:NO /OPT:REF /OPT:ICF /SUBSYSTEM:WINDOWS")
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libtango.lib
                                                $ENV{TANGO_PATH}/lib/omniORB4.lib
                                                $ENV{TANGO_PATH}/lib/omniDynamic4.lib
                                                $ENV{TANGO_PATH}/lib/COS4.lib
                                                $ENV{TANGO_PATH}/lib/omnithread.lib
                                                $ENV{TANGO_PATH}/bin/msvcstub.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC ws2_32 mswsock advapi32 comctl32 odbc32)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/pthreadVC2.lib)
  if(MSVC90)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v90-mt-s-4_0_5.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/libboost_python-vc90-mt-s-1_64.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{PYTHONPATH}/libs/python27.lib)
  elseif(MSVC10)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v100-mt-s-4_0_5.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/libboost_python3-vc100-mt-s-1_64.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{PYTHONPATH}/libs/python34.lib)
  elseif(MSVC12)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v120-mt-s-4_0_5.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/libboost_python3-vc120-mt-s-1_64.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{PYTHONPATH}/libs/python34.lib)
  elseif(MSVC14)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v140-mt-s-4_0_5.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/libboost_python3-vc140-mt-s-1_64.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{PYTHONPATH}/libs/python35.lib)
  endif(MSVC90)
  
else(WIN32)
  add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TANGO_PKG_INCLUDE_DIRS})
  target_compile_options(${PROJECT_NAME} PUBLIC -std=c++11)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${TANGO_PKG_CFLAGS_OTHER})
  link_directories(${TANGO_PKG_LIBRARY_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${TANGO_PKG_LIBRARIES})
endif(WIN32)
