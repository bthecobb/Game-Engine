@echo off
setlocal
set "BUILD_DIR=%~dp0build"
set "RELEASE_DIR=%BUILD_DIR%\Release"
set "EXE_DIR=%RELEASE_DIR%\RelWithDebInfo"
set "PATH=%RELEASE_DIR%;%PATH%"

echo Launching Full3DGame_DX12...
if exist "%EXE_DIR%\Full3DGame_DX12_Debug3_Run2.exe" (
    cd /d "%EXE_DIR%"
    start "" "Full3DGame_DX12_Debug3_Run2.exe"
) else (
    echo Error: Executable not found in %EXE_DIR%
    pause
)
endlocal
