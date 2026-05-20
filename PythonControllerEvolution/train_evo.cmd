@echo off
setlocal enabledelayedexpansion

:: Usage: train_evo.cmd [model-name] [level-set] [episodes]
::   model-name  saved as ..\PythonController\checkpoints\ppo_<name>.pt  (default: evo)
::   level-set   subfolder under Mario-AI-Framework\levels\              (default: o)
::   episodes    number of episodes                                       (default: 100)

set MODEL_NAME=%~1
if "%MODEL_NAME%"=="" set MODEL_NAME=evo

set LEVEL_SET=%~2
if "%LEVEL_SET%"=="" set LEVEL_SET=o

set EPISODES=%~3
if "%EPISODES%"=="" set EPISODES=100

set SCRIPT_DIR=%~dp0
set FRAMEWORK_DIR=%SCRIPT_DIR%..\Mario-AI-Framework
set LEVELS_DIR=%FRAMEWORK_DIR%\levels\%LEVEL_SET%
set CHECKPOINT=%SCRIPT_DIR%..\PythonController\checkpoints\ppo_%MODEL_NAME%.pt
set STATS=%SCRIPT_DIR%..\PythonController\checkpoints\ppo_%MODEL_NAME%_stats.jsonl
set TB_DIR=%SCRIPT_DIR%..\PythonController\runs\%MODEL_NAME%
set PYTHON=%SCRIPT_DIR%..\.venv\Scripts\python.exe

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

if "%LEVEL_PATHS%"=="" (
    echo ERROR: No .txt level files found in %LEVELS_DIR%
    exit /b 1
)

echo ================================================
echo  Mario PPO Evolution Trainer
echo ================================================
echo  Model:       %MODEL_NAME%
echo  Level set:   %LEVEL_SET%
echo  Episodes:    %EPISODES%
echo  Checkpoint:  %CHECKPOINT%
echo  TensorBoard: %TB_DIR%
echo  GPU:         RTX 4080 (AMP enabled)
echo ================================================
echo.

:: Start evolution Python controller in a separate window
set PS1=%TEMP%\start_evo_train.ps1
echo $p = Start-Process -FilePath '%PYTHON%' -ArgumentList @('%SCRIPT_DIR%controller_evo.py','--model-path','%CHECKPOINT%','--stats-path','%STATS%','--tensorboard-dir','%TB_DIR%','--save-every','10') -PassThru -WindowStyle Normal > "%PS1%"
echo Write-Host "Evolution controller PID: $($p.Id)" >> "%PS1%"
powershell -ExecutionPolicy Bypass -File "%PS1%"

:: Give the Python server time to start
ping 127.0.0.1 -n 8 >nul

:: Compile and run Java
pushd "%FRAMEWORK_DIR%"

echo Compiling Java...
javac -cp src src\mff\python\PythonControllerMain.java
if errorlevel 1 (
    echo ERROR: Java compilation failed.
    popd
    exit /b 1
)

echo Starting %EPISODES% episodes on level set '%LEVEL_SET%'...
java -cp src mff.python.PythonControllerMain 127.0.0.1 5050 "%LEVEL_PATHS%" 200 0 false %EPISODES% 30

popd
