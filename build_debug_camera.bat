@echo off
echo.
echo ===============================================
echo  Building CudaGame with Robust Camera System
echo ===============================================
echo.

REM Navigate to build directory
cd /d "C:\Users\Brandon\CudaGame\build\Release"

REM Clean and build the project
echo Cleaning previous build...
cmake --build . --target clean

echo.
echo Building with robust player position management...
cmake --build . --config Release --target EnhancedGameMain_Full3D

echo.
echo ===============================================
echo  Build Complete!
echo ===============================================
echo.
echo Controls for Testing:
echo - K key: Toggle between KINEMATIC/DYNAMIC modes
echo - WASD: Move player (in DYNAMIC mode)
echo - TAB: Toggle mouse capture
echo - Camera modes: 1, 2, 3
echo.
echo Expected Behavior:
echo - KINEMATIC mode: Camera stable, player position fixed
echo - DYNAMIC mode: Camera follows player movement
echo - No camera flickering or drift in either mode
echo - Console shows detailed debugging output
echo.

REM Check if build succeeded
if exist "EnhancedGameMain_Full3D.exe" (
    echo Build successful! Ready to run EnhancedGameMain_Full3D.exe
    echo.
    set /p run="Run the game now? (y/n): "
    if /i "!run!"=="y" (
        echo.
        echo Starting CudaGame...
        echo.
        EnhancedGameMain_Full3D.exe
    )
) else (
    echo.
    echo Build failed! Check the output above for errors.
    echo Make sure all dependencies are properly configured.
)

echo.
pause
