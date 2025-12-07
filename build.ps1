# GigaLearnCPP Build Script for Visual Studio 2026 with CUDA 13.0
# This script configures and builds the project with CUDA support

param(
    [string]$BuildType = "RelWithDebInfo",
    [switch]$Clean = $false,
    [switch]$Configure = $true,
    [switch]$Build = $true,
    [string]$CudaArch = "86"
)

# Configuration
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "out\build\x64-RelWithDebInfo"
$CudaPath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
$LibTorchPath = "C:\Giga\GigaLearnCPP-Leak\libtorch"
$VSPath = "C:\Program Files\Microsoft Visual Studio\18\Community"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GigaLearnCPP Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "Build Directory: $BuildDir" -ForegroundColor Yellow
Write-Host "CUDA Path: $CudaPath" -ForegroundColor Yellow
Write-Host "LibTorch Path: $LibTorchPath" -ForegroundColor Yellow
Write-Host "Visual Studio Path: $VSPath" -ForegroundColor Yellow
Write-Host ""

# Verify paths exist
if (-not (Test-Path $CudaPath)) {
    Write-Host "ERROR: CUDA path not found: $CudaPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "ERROR: LibTorch path not found: $LibTorchPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VSPath)) {
    Write-Host "ERROR: Visual Studio path not found: $VSPath" -ForegroundColor Red
    exit 1
}

# Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
    Write-Host "Build directory cleaned." -ForegroundColor Green
}

# Create build directory if it doesn't exist
if (-not (Test-Path $BuildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# Set environment variables (forward slashes avoid escaping issues inside CMake)
$env:CUDA_PATH = $CudaPath
$env:CUDA_HOME = $CudaPath
$env:CUDA_BIN_PATH = "$CudaPath/bin"
$env:PATH = "$CudaPath/bin;$env:PATH"

# Initialize Visual Studio environment
Write-Host "Initializing Visual Studio 2026 environment..." -ForegroundColor Yellow
$VsDevCmd = Join-Path $VSPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
    Write-Host "ERROR: VsDevCmd.bat not found at: $VsDevCmd" -ForegroundColor Red
    exit 1
}

# Run VsDevCmd in a cmd shell and capture environment variables
$tempFile = [System.IO.Path]::GetTempFileName()
$cmdExe = $env:ComSpec
if (-not $cmdExe -or -not (Test-Path $cmdExe)) {
    $cmdExe = "C:\Windows\System32\cmd.exe"
}
& $cmdExe /c "`"$VsDevCmd`" -arch=amd64 && set" > $tempFile

# Parse and set environment variables
Get-Content $tempFile | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path "env:$name" -Value $value -ErrorAction SilentlyContinue
    }
}
Remove-Item $tempFile

Write-Host "Visual Studio environment initialized." -ForegroundColor Green
Write-Host ""

# Configure with CMake
if ($Configure) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Configuring CMake..." -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $CmakeArgs = @(
        "-S", $ProjectRoot,
        "-B", $BuildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe",
        "-DCUDA_TOOLKIT_ROOT_DIR=$CudaPath",
        "-DCMAKE_PREFIX_PATH=$LibTorchPath",
        "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    )
    
    Write-Host "Running: cmake $($CmakeArgs -join ' ')" -ForegroundColor Yellow
    Write-Host ""
    
    & cmake @CmakeArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: CMake configuration failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "CMake configuration completed successfully." -ForegroundColor Green
    Write-Host ""
}

# Build with CMake
if ($Build) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building project..." -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $BuildArgs = @(
        "--build", $BuildDir,
        "--config", $BuildType,
        "--parallel"
    )
    
    Write-Host "Running: cmake $($BuildArgs -join ' ')" -ForegroundColor Yellow
    Write-Host ""
    
    & cmake @BuildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "Build completed successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build script finished successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output directory: $BuildDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  .\build.ps1                    # Configure and build" -ForegroundColor Gray
Write-Host "  .\build.ps1 -Clean             # Clean, configure, and build" -ForegroundColor Gray
Write-Host "  .\build.ps1 -Configure:$false  # Build only (skip configure)" -ForegroundColor Gray
Write-Host "  .\build.ps1 -Build:$false      # Configure only (skip build)" -ForegroundColor Gray
