# Install script for directory: /home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/FilterConfig.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/FilterPainter.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/FilterStack.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/FilterTarget.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/LowPassFilter.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/Oracle.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/Streamline.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/VectorField.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/vtkTurkBanksStreamlineFilter.h"
    "/home/alba/projects/TurkBanksPlugin/Plugin/TurkBanksStreamlineFilterSources/TurkBanksStreamlineFilterSourcesModule.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/vtk/hierarchy/TurkBanksStreamlineFilter" TYPE FILE RENAME "TurkBanksStreamlineFilterSources-hierarchy.txt" FILES "/home/alba/projects/TurkBanksPlugin/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/vtk/hierarchy/TurkBanksStreamlineFilter/TurkBanksStreamlineFilterSources-hierarchy.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xruntimex" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so"
         RPATH "$ORIGIN:$ORIGIN/../../../")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter" TYPE SHARED_LIBRARY FILES "/home/alba/projects/TurkBanksPlugin/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so"
         OLD_RPATH "/home/alba/paraview_build/lib:"
         NEW_RPATH "$ORIGIN:$ORIGIN/../../../")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/paraview-5.12/plugins/TurkBanksStreamlineFilter/libTurkBanksStreamlineFilterSources.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

