@echo off
setlocal

rem Build MiniOps with OpenMP using LLVM clang++ + libomp.
rem TDM-GCC 10.3 ships omp.h but no libgomp, so the default toolchain
rem cannot link an OpenMP binary. CMake's FindOpenMP also needs help on
rem Windows: the runtime is libomp.lib, not omp.lib.

set "ROOT=%~dp0.."
set "BUILD_DIR=%ROOT%\build-omp"

if not defined CXX set "CXX=clang++.exe"

if exist "%CXX%" (
    set "CXX_PATH=%CXX%"
    goto :found
)

for /f "delims=" %%I in ('where %CXX% 2^>nul') do (
    set "CXX_PATH=%%I"
    goto :found
)

echo [ERROR] clang++ not found. Install LLVM or set CXX to clang++.exe.
exit /b 1

:found
for %%I in ("%CXX_PATH%") do set "LLVM_DIR=%%~dpI.."
set "LIBOMP=%LLVM_DIR%\lib\libomp.lib"
if not exist "%LIBOMP%" (
    echo [ERROR] libomp.lib not found: %LIBOMP%
    exit /b 1
)

echo Using compiler: %CXX_PATH%
echo Using runtime:  %LIBOMP%

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER="%CXX_PATH%" ^
    -DMINIOPS_ENABLE_OPENMP=ON ^
    -DOpenMP_CXX_FLAGS=-fopenmp ^
    -DOpenMP_CXX_LIB_NAMES=libomp ^
    -DOpenMP_libomp_LIBRARY="%LIBOMP%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\test_matmul.exe"
exit /b %errorlevel%
