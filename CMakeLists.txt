cmake_minimum_required(VERSION 3.0.2)
set(CMP0048 NEW)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        project(PyTango_d LANGUAGES CXX
                VERSION 1.0)
else()
        project(PyTango LANGUAGES CXX
                VERSION 1.0)
endif()
#need to define the version of the library
set(MAJOR_VERSION "9")
set(MINOR_VERSION "2")
set(PATCH_VERSION "5")
#convenient versions
set(LIBRARY_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
IF(NOT WIN32)
include(FindPkgConfig)

pkg_search_module(TANGO_PKG REQUIRED tango)
ENDIF()

IF (WIN32)
add_definitions(-DWIN32)
add_definitions(-DTANGO_HAS_DLL)
add_definitions(-DLOG4TANGO_HAS_DLL)
add_definitions(-D_WINSOCKAPI_)
add_definitions(-DNPY_NO_DEPRECATED_API)
add_definitions(-DBOOST_PYTHON_STATIC_LIB)
add_compile_options(/bigobj)
IF(CMAKE_CL_64)
add_definitions(-D_64BITS)
IF (MSVC14)
add_definitions(-D_TIMERS_T_)
ENDIF()
ELSE()
add_definitions(-DJPG_USE_ASM)
ENDIF()
ENDIF()
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
        ext/server/dserver.cpp
        ext/server/encoded_attribute.cpp
        ext/enums.cpp
        ext/event_data.cpp
        ext/exception.cpp
        ext/from_py.cpp
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

IF(WIN32)

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
  
  add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS}) 
  SET_TARGET_PROPERTIES( ${PROJECT_NAME}
         PROPERTIES
         SUFFIX ".pyd" )
         
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
add_definitions(-D_DEBUG)
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        CONFIGURATIONS Debug)
install(FILES Debug/${PROJECT_NAME}.lib DESTINATION bin)
else()
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
endif()
  set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS "/bigobj")
  set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS "/force:multiple")
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/tangod.lib
                                                $ENV{TANGO_PATH}/bin/omniORB4_rtd.lib
                                                $ENV{TANGO_PATH}/bin/omniDynamic4_rtd.lib
                                                $ENV{TANGO_PATH}/bin/COS4_rtd.lib
                                                $ENV{TANGO_PATH}/bin/omnithread_rtd.lib
                                                $ENV{TANGO_PATH}/bin/msvcstubd.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC ws2_32 mswsock advapi32 comctl32 odbc32)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/pthreadVC2.lib)
  IF(MSVC90)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v90-mt-gd-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc90-mt-gd-1_64.lib)
  ELSEIF(MSVC10)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v100-mt-gd-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc100-mt-gd-1_64.lib)
  ELSEIF(MSVC12)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v120-mt-gd-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc120-mt-gd-1_64.lib)
  ELSEIF(MSVC14)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v140-mt-gd-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc140-mt-gd-1_64.lib)
  ENDIF()
  else()
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/tango.lib
                                                $ENV{TANGO_PATH}/bin/omniORB4_rt.lib
                                                $ENV{TANGO_PATH}/bin/omniDynamic4_rt.lib
                                                $ENV{TANGO_PATH}/bin/COS4_rt.lib
                                                $ENV{TANGO_PATH}/bin/omnithread_rt.lib
                                                $ENV{TANGO_PATH}/bin/msvcstub.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC ws2_32 mswsock advapi32 comctl32 odbc32)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/pthreadVC2.lib)
  IF(MSVC90)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v90-mt-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc90-mt-1_64.lib)
  ELSEIF(MSVC10)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v100-mt-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc100-mt-1_64.lib)
  ELSEIF(MSVC12)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v120-mt-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc120-mt-1_64.lib)
  ELSEIF(MSVC14)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{TANGO_PATH}/lib/libzmq-v140-mt-4_1_7.lib)
    target_link_libraries(${PROJECT_NAME} PUBLIC $ENV{BOOST_LIBS}/boost_python-vc140-mt-1_64.lib)
  ENDIF()
  endif()
  
ELSE()
  add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TANGO_PKG_INCLUDE_DIRS})
  target_compile_options(${PROJECT_NAME} PUBLIC -std=c++11)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${TANGO_PKG_CFLAGS_OTHER})
  link_directories(${TANGO_PKG_LIBRARY_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${TANGO_PKG_LIBRARIES})
ENDIF()
IF(WIN32)
IF(MSVC90)
      if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        IF(CMAKE_CL_64)
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_v90_x64_debug)
        ELSE()
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_v90_x86_debug)
        ENDIF()
      else()
        IF(CMAKE_CL_64)
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_v90_x64)
        ELSE()
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_v90_x86)
        ENDIF()
      endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ELSE()
      IF(CMAKE_CL_64)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")  
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_${CMAKE_VS_PLATFORM_TOOLSET}_x64_debug)
        else()
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_${CMAKE_VS_PLATFORM_TOOLSET}_x64)
        endif()
      ELSE()
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_${CMAKE_VS_PLATFORM_TOOLSET}_x86_debug)
        else()
          set(CPACK_PACKAGE_FILE_NAME PyTango_${LIBRARY_VERSION}_${CMAKE_VS_PLATFORM_TOOLSET}_x86)
        endif()
      ENDIF()
    ENDIF(MSVC90)
  set (CPACK_PACKAGE_VENDOR "www.tango-controls.org")
  set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tango - Connecting Things Together")
  set (CPACK_PACKAGE_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
  set (CPACK_PACKAGE_VERSION_MAJOR ${MAJOR_VERSION})
  set (CPACK_PACKAGE_VERSION_MINOR ${MINOR_VERSION})
  set (CPACK_PACKAGE_VERSION_PATCH ${PATCH_VERSION})
  
  set(CPACK_NSIS_HELP_LINK "http://www.tango-controls.org")
  set(CPACK_NSIS_URL_INFO_ABOUT "http://www.tango-controls.org")
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  set(CPACK_NSIS_MENU_LINKS
      "http://tango-controls.readthedocs.io/en/latest/" "Tango Doc")
ENDIF()
#include(CPack)
