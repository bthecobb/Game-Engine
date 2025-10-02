# External dependencies configuration

include(FetchContent)

# GLM
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/glm/CMakeLists.txt")
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(glm)
else()
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/glm)
endif()

# GLFW
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/glfw/CMakeLists.txt")
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(glfw)
else()
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/glfw)
endif()

# Vulkan Headers
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/vulkan/CMakeLists.txt")
    FetchContent_Declare(
        vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(vulkan_headers)
else()
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/vulkan)
endif()

# Function to ensure all dependencies are available
function(ensure_dependencies)
    # Check GLM
    if(NOT TARGET glm::glm)
        message(FATAL_ERROR "GLM not found")
    endif()
    
    # Check GLFW
    if(NOT TARGET glfw)
        message(FATAL_ERROR "GLFW not found")
    endif()
    
    # Check Vulkan
    if(NOT TARGET Vulkan::Headers)
        message(FATAL_ERROR "Vulkan Headers not found")
    endif()
endfunction()