@echo off
REM ---------------------------------------------------------------
REM build.bat - compile the project's .tex files (currently just
REM the data-integrity disclaimer) using TeX Live inside WSL
REM Ubuntu, from a Windows shell.
REM
REM Same pattern as semester-2\chronicles\build.bat — delegates to
REM a .sh on the Linux side so the real pdflatex invocation lives
REM in one place.
REM
REM Usage (no args — compiles plots\data_integrity\disclaimer.tex):
REM     build.bat
REM
REM Usage (with explicit .tex path):
REM     build.bat "path\to\file.tex"
REM
REM Output lands in a build\ subfolder beside the source, never
REM overwriting the source directory.
REM ---------------------------------------------------------------

setlocal

if "%~1"=="" (
    set "TARGET=%~dp0plots\data_integrity\disclaimer.tex"
) else (
    set "TARGET=%~f1"
)

if not exist "%TARGET%" (
    echo Not found: %TARGET%
    exit /b 1
)

REM Translate the Windows paths into the /mnt/c form WSL understands.
for /f "usebackq delims=" %%P in (`wsl.exe wslpath -a "%TARGET%"`) do set "TEXPATH=%%P"
for /f "usebackq delims=" %%S in (`wsl.exe wslpath -a "%~dp0wsl-build.sh"`) do set "SCRIPT=%%S"

wsl.exe -d Ubuntu bash "%SCRIPT%" "%TEXPATH%"

endlocal
