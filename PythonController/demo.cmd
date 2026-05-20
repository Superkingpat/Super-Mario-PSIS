@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: demo.cmd ^<model-name^> [level-set] [sessions]
    echo   model-name  Model to pre-train  ^(demos saved to demos\^<name^>_demos.jsonl^)
    echo   level-set   Level folder  ^(default: o^)
    echo   sessions    Number of episodes to play  ^(default: 10^)
    exit /b 1
)

set MODEL_NAME=%~1
set LEVEL_SET=%~2
if "%LEVEL_SET%"=="" set LEVEL_SET=o
set SESSIONS=%~3
if "%SESSIONS%"=="" set SESSIONS=10

set SCRIPT_DIR=%~dp0
set FRAMEWORK_DIR=%SCRIPT_DIR%..\Mario-AI-Framework
set LEVELS_DIR=%FRAMEWORK_DIR%\levels\%LEVEL_SET%
set DEMO_PATH=%SCRIPT_DIR%demos\%MODEL_NAME%_demos.jsonl
set MODEL_PATH=%SCRIPT_DIR%checkpoints\ppo_%MODEL_NAME%.pt
set PYTHON=%SCRIPT_DIR%..\\.venv\Scripts\python.exe

if not exist "%LEVELS_DIR%" (
    echo ERROR: Level set not found: %LEVELS_DIR%
    exit /b 1
)

:: Build level paths
set LEVEL_PATHS=
for %%F in ("%LEVELS_DIR%\*.txt") do (
    if "!LEVEL_PATHS!"=="" (
        set LEVEL_PATHS=./levels/%LEVEL_SET%/%%~nxF
    ) else (
        set LEVEL_PATHS=!LEVEL_PATHS!;./levels/%LEVEL_SET%/%%~nxF
    )
)

echo Model:    %MODEL_NAME%
echo Levels:   %LEVEL_SET%
echo Sessions: %SESSIONS%
echo Demos:    %DEMO_PATH%
echo.
echo Controls: Arrow keys to move, Space to jump, Shift to run
echo.

:: Write and run a small PS1 to avoid inline escaping issues with spaces in paths
set PS1=%TEMP%\start_demo.ps1
echo $p = Start-Process -FilePath '%PYTHON%' -ArgumentList @('%SCRIPT_DIR%human_demo.py','--demo-path','%DEMO_PATH%','--sessions','%SESSIONS%') -PassThru -WindowStyle Normal > "%PS1%"
echo Write-Host "Demo server PID: $($p.Id)" >> "%PS1%"
powershell -ExecutionPolicy Bypass -File "%PS1%"

:: Give the server time to bind the port
timeout /t 5 /nobreak >nul

:: Compile Java
pushd "%FRAMEWORK_DIR%"
javac -cp src src\mff\python\PythonControllerMain.java
if errorlevel 1 ( echo Java compile failed & popd & exit /b 1 )

:: Run with visuals ON at 24 fps so the human can play comfortably
java -cp src mff.python.PythonControllerMain 127.0.0.1 5050 "%LEVEL_PATHS%" 200 0 true %SESSIONS% 60 24
popd

echo.
echo Done! Now train the model on your demos:
echo   %PYTHON% "%SCRIPT_DIR%train_bc.py" --demo-path "%DEMO_PATH%" --model-path "%MODEL_PATH%"
echo.
echo Then fine-tune with PPO:
echo   train.cmd %MODEL_NAME% %LEVEL_SET% 1000
