if(TARGET CLI11::CLI11)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)