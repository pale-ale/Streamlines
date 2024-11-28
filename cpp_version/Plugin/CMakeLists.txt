set(BUILD_SHARED_LIBS TRUE)

paraview_add_plugin(TurkBanksStreamlineFilter
  REQUIRED_ON_CLIENT
  REQUIRED_ON_SERVER
  VERSION "1.0"
  MODULES
    TurkBanksStreamlineFilterSources
  MODULE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/TurkBanksStreamlineFilterSources/vtk.module"
)

option(BUILD_TESTING "Build Plugin Testing" OFF)
if (BUILD_TESTING AND BUILD_SHARED_LIBS)
  add_subdirectory(Testing)
endif()
