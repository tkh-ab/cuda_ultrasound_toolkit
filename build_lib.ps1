<#
.SYNOPSIS
    Builds the CUDA Toolkit library using MSBuild.

.PARAMETER OutputDir
    Output location. Default: .\library_output.
	Alias: o

.PARAMETER Configuration
    Build configuration [Debug|Release].  Default: Release.
	Alias: c

.PARAMETER Clean
    If supplied, runs a Clean before the Build target.

.PARAMETER Help
	Show this help message.
	Alias: h

.EXAMPLE
    .\build_lib.ps1 -o .\build -c Debug
#>

[CmdletBinding()]
param (
	[Alias("h")]
	[switch]$Help,

	[Alias("o")]
    [ValidateNotNullOrEmpty()]
    [string]$OutputDir = ".\library_output",

	[Alias("c")]
    [ValidateNotNullOrEmpty()]
    [string]$Configuration = "Release",

    [switch]$Clean
)

if ($Help) {
    # Show the full help (synopsis, parameters, examples)
    Get-Help -Detailed -ErrorAction SilentlyContinue $PSCommandPath
    return
}

$PublicDir = "$PSScriptRoot\cuda_toolkit\src\public"
$MatlabDir = "$PublicDir\matlab"
$HeaderOutputDir = "$OutputDir\include"
$MatlabOutputDir = "$OutputDir\matlab"

$Project = "$PSScriptRoot\cuda_toolkit\cuda_toolkit.vcxproj"
$Platform = "x64"

# ---- Locate MSBuild ---------------------------------------------------------
$vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at $vswhere; install Visual Studio 2017+ or add vswhere to PATH."
}

$msbuildPath = & $vswhere -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe | Select-Object -First 1
if (-not $msbuildPath) { throw "MSBuild.exe not found by vswhere." }

# ---- Prepare output directory ----------------------------------------------
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Trailing backslash is required so MSBuild treats it as a directory, not a file.
$normalizedOutDir = ([System.IO.Path]::GetFullPath($OutputDir)) + "\bin\"

# ---- Compose MSBuild arguments ---------------------------------------------
$targets = if ($Clean) { "Clean;Build" } else { "Build" }

$msbuildArgs = @(
    $Project,
    "/t:$targets",
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform",
    "/p:OutDir=$normalizedOutDir",     # put binaries where you asked
    "/m",                              # parallel build
    "/nr:false",                       # no node-reuse (friendlier in scripts)
    "/v:minimal"                       # quieter output; change to detailed if needed
)

Write-Host "Building $Configuration"
& $msbuildPath @msbuildArgs

$build_exit_code = $LASTEXITCODE

if(-not (Test-Path $HeaderOutputDir)) {
	New-Item -ItemType Directory -Path $HeaderOutputDir | Out-Null
}

if(-not (Test-Path $MatlabOutputDir)) {
	New-Item -ItemType Directory -Path $MatlabOutputDir | Out-Null
}

Copy-Item -Path "$PublicDir\*.h" -Destination "$HeaderOutputDir" -Force

if(Test-Path $MatlabDir) {
	Copy-Item -Path "$MatlabDir\*" -Destination "$MatlabOutputDir" -Force
}

if ($build_exit_code -ne 0) {
	Write-Error "MSBuild failed with exit code $build_exit_code."
}
exit $process.ExitCode
