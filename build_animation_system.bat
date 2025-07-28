@echo off
echo 🎮 Enhanced Animation System Build Script
echo ==========================================

REM Set build configuration
set CONFIG=Release
set PLATFORM=x64

REM Create build directories
if not exist "build" mkdir build
if not exist "build\animation" mkdir build\animation
if not exist "bin" mkdir bin

echo 📁 Build directories created

REM Check for required dependencies
echo 🔍 Checking dependencies...

REM Check for Visual Studio
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Visual Studio compiler not found in PATH
    echo Please run this from a Visual Studio Developer Command Prompt
    pause
    exit /b 1
)

REM Check for GLM (OpenGL Mathematics library)
if not exist "third_party\glm\glm\glm.hpp" (
    echo ⬇️ GLM not found, downloading...
    if not exist "third_party" mkdir third_party
    cd third_party
    git clone https://github.com/g-truc/glm.git
    cd ..
)

echo ✅ Dependencies checked

REM Compilation flags
set INCLUDES=-I"src" -I"third_party\glm" -I"include"
set CFLAGS=/std:c++17 /O2 /W3 /EHsc /D"_CRT_SECURE_NO_WARNINGS"
set LIBS=

echo 🔨 Compiling Animation System...

REM Compile Animation System Core
echo   Compiling AnimationSystem.cpp...
cl %CFLAGS% %INCLUDES% /c src\AnimationSystem.cpp /Fo:build\animation\AnimationSystem.obj
if %ERRORLEVEL% NEQ 0 goto :error

echo   Compiling AnimationUtils.cpp...
cl %CFLAGS% %INCLUDES% /c src\AnimationUtils.cpp /Fo:build\animation\AnimationUtils.obj
if %ERRORLEVEL% NEQ 0 goto :error

echo   Compiling RhythmFeedbackSystem.cpp...
cl %CFLAGS% %INCLUDES% /c src\RhythmFeedbackSystem.cpp /Fo:build\animation\RhythmFeedbackSystem.obj
if %ERRORLEVEL% NEQ 0 goto :error

echo   Compiling AnimationDemo.cpp...
cl %CFLAGS% %INCLUDES% /c src\AnimationDemo.cpp /Fo:build\animation\AnimationDemo.obj
if %ERRORLEVEL% NEQ 0 goto :error

echo 🔗 Linking executable...

REM Link the demo executable
cl build\animation\*.obj /Fe:bin\AnimationDemo.exe %LIBS%
if %ERRORLEVEL% NEQ 0 goto :error

echo ✅ Build completed successfully!
echo 
echo 📦 Output files:
echo   bin\AnimationDemo.exe - Interactive demonstration
echo 
echo 🚀 To run the demo:
echo   cd bin
echo   AnimationDemo.exe
echo 
echo 🎯 Features included:
echo   • Complete movement animation cycles
echo   • Real-time rhythm analysis and beat detection  
echo   • Animation synchronization with audio beats
echo   • Procedural animation enhancements
echo   • Event-driven feedback systems
echo   • Adaptive movement and visual modulation
echo 
echo Build completed at %TIME% on %DATE%
goto :end

:error
echo ❌ Build failed with error code %ERRORLEVEL%
pause
exit /b %ERRORLEVEL%

:end
echo 
echo 🎉 Ready to run Enhanced Animation System Demo!
pause
