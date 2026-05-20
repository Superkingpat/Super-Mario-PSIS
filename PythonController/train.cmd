@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: train.cmd ^<model-name^> [level-set] [sessions]
    echo   model-name  Name for the model  ^(saved as checkpoints\ppo_^<name^>.pt^)
    echo   level-set   Subfolder under Mario-AI-Framework\levels\  ^(default: original^)
    echo   sessions    Number of episodes to run  ^(default: 100000^)
    exit /b 1
)

set MODEL_NAME=%~1
set LEVEL_SET=%~2
if "%LEVEL_SET%"=="" set LEVEL_SET=original
set SESSIONS=%~3
if "%SESSIONS%"=="" set SESSIONS=100000

set SCRIPT_DIR=%~dp0
set FRAMEWORK_DIR=%SCRIPT_DIR%..\Mario-AI-Framework
set LEVELS_DIR=%FRAMEWORK_DIR%\levels\%LEVEL_SET%
set CHECKPOINT=%SCRIPT_DIR%checkpoints\ppo_%MODEL_NAME%.pt
set STATS=%SCRIPT_DIR%checkpoints\ppo_%MODEL_NAME%_stats.jsonl
set TB_DIR=%SCRIPT_DIR%runs\%MODEL_NAME%
set PYTHON=%SCRIPT_DIR%..\\.venv\Scripts\python.exe

if not exist "%LEVELS_DIR%" (
    echo ERROR: Level set not found: %LEVELS_DIR%
    exit /b 1
)

:: Build semicolon-separated level paths relative to Mario-AI-Framework\
set LEVEL_PATHS=
for %%F in ("%LEVELS_DIR%\*.txt") do (
    if "!LEVEL_PATHS!"=="" (
        set LEVEL_PATHS=./levels/%LEVEL_SET%/%%~nxF
    ) else (
        set LEVEL_PATHS=!LEVEL_PATHS!;./levels/%LEVEL_SET%/%%~nxF
    )
)

echo Model:       %MODEL_NAME%
echo Level set:   %LEVEL_SET%
echo Sessions:    %SESSIONS%
echo Checkpoint:  %CHECKPOINT%
echo TensorBoard: %TB_DIR%
echo.

:: Start Python controller in a separate window via PowerShell to avoid escaping issues
set PS1=%TEMP%\start_train.ps1
echo $p = Start-Process -FilePath '%PYTHON%' -ArgumentList @('%SCRIPT_DIR%controller.py','--model-path','%CHECKPOINT%','--stats-path','%STATS%','--tensorboard-dir','%TB_DIR%','--save-every','10') -PassThru -WindowStyle Normal > "%PS1%"
echo Write-Host "Python controller PID: $($p.Id)" >> "%PS1%"
powershell -ExecutionPolicy Bypass -File "%PS1%"

:: Wait for the Python server to be ready (ping works even when stdin is redirected)
ping 127.0.0.1 -n 8 >nul

:: Compile and run Java from the framework directory
pushd "%FRAMEWORK_DIR%"

echo Compiling Java...
javac -cp src src\mff\python\PythonControllerMain.java
if errorlevel 1 (
    echo ERROR: Java compilation failed.
    popd
    exit /b 1
)

echo Starting training...
java -cp src mff.python.PythonControllerMain 127.0.0.1 5050 "%LEVEL_PATHS%" 200 0 false %SESSIONS% 30

popd
