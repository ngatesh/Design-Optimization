@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-LD1D5M7" (taskkill /f /pid 4028)
if /i "%LOCALHOST%"=="DESKTOP-LD1D5M7" (taskkill /f /pid 13280)

del /F cleanup-ansys-DESKTOP-LD1D5M7-13280.bat
