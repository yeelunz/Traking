@echo off
setlocal ENABLEDELAYEDEXPANSION

rem Change directory to repo root
cd /d "%~dp0"

rem Allow overriding the results root, port, and conda env via arguments
set "RESULTS_DIR=%~1"
if "%RESULTS_DIR%"=="" set "RESULTS_DIR=results"

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8008"

set "CONDA_ENV=%~3"
if "%CONDA_ENV%"=="" (
	if defined CONDA_DEFAULT_ENV set "CONDA_ENV=%CONDA_DEFAULT_ENV%"
)
if "%CONDA_ENV%"=="" set "CONDA_ENV=py311"

set "DEFAULT_CONDA_ROOT=%USERPROFILE%\anaconda3"
if not "%CONDA_ROOT%"=="" set "DEFAULT_CONDA_ROOT=%CONDA_ROOT%"
if exist "%DEFAULT_CONDA_ROOT%\Scripts\conda.exe" set "CONDA_EXE=%DEFAULT_CONDA_ROOT%\Scripts\conda.exe"
if exist "%DEFAULT_CONDA_ROOT%\python.exe" set "CONDA_BASE_PY=%DEFAULT_CONDA_ROOT%\python.exe"

set "CONDA_CMD="
if defined CONDA_EXE (
	set "CONDA_CMD=%CONDA_EXE%"
) else (
	where conda >nul 2>&1 && set "CONDA_CMD=conda"
)

set "HOST=0.0.0.0"
set "BROWSER_HOST=0.0.0.0"
set "VIEWER_REQUIREMENTS=tools\schedule_web_viewer\requirements.txt"
if "%HOST%"=="0.0.0.0" (
	set "BROWSER_HOST=127.0.0.1"
) else (
	set "BROWSER_HOST=%HOST%"
)

set "CONDA_ACTIVATE_BAT="
set "USE_CONDA_ACTIVATE=0"

set "PYTHON_CMD="
set "USE_CONDA_RUN=0"

if exist "%DEFAULT_CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ACTIVATE_BAT=%DEFAULT_CONDA_ROOT%\Scripts\activate.bat"

if not "%CONDA_ENV%"=="" (
	if not "%CONDA_ACTIVATE_BAT%"=="" (
		set "USE_CONDA_ACTIVATE=1"
	) else if not "%CONDA_CMD%"=="" (
		set "USE_CONDA_RUN=1"
	) else (
		echo [WARN] Requested Conda env %CONDA_ENV% but conda.exe was not found. Set CONDA_ROOT or update PATH.
	)
)

if "%USE_CONDA_RUN%"=="0" if "%USE_CONDA_ACTIVATE%"=="0" if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if "%USE_CONDA_RUN%"=="0" if "%USE_CONDA_ACTIVATE%"=="0" if "%PYTHON_CMD%"=="" if defined CONDA_BASE_PY set "PYTHON_CMD=%CONDA_BASE_PY%"
if "%USE_CONDA_RUN%"=="0" if "%USE_CONDA_ACTIVATE%"=="0" if "%PYTHON_CMD%"=="" (
	where python >nul 2>&1 && set "PYTHON_CMD=python"
)
if "%USE_CONDA_RUN%"=="0" if "%USE_CONDA_ACTIVATE%"=="0" if "%PYTHON_CMD%"=="" (
	where py >nul 2>&1 && set "PYTHON_CMD=py -3"
)
if "%USE_CONDA_RUN%"=="0" if "%USE_CONDA_ACTIVATE%"=="0" if "%PYTHON_CMD%"=="" (
	echo [ERROR] Unable to find a Python interpreter. Ensure python is installed or create .venv.
	echo.
	pause
	goto :EOF
)

if "%USE_CONDA_ACTIVATE%"=="1" (
	echo Using interpreter: python (via conda activate)
	echo   (conda env: %CONDA_ENV%)
) else if "%USE_CONDA_RUN%"=="1" (
	echo Using interpreter: %CONDA_CMD% (via conda run)
	echo   (conda env: %CONDA_ENV%)
) else (
	echo Using interpreter: %PYTHON_CMD%
)
echo Launching Schedule Results Viewer
@echo   Results root: %RESULTS_DIR%


rem Ensure viewer-specific dependencies exist before launching
set "NEED_VIEWER_DEPS=0"
if exist "%VIEWER_REQUIREMENTS%" (
	set "NEED_VIEWER_DEPS=1"
	if "%USE_CONDA_ACTIVATE%"=="1" (
		call :RunInCondaEnv python -c "import fastapi, uvicorn" >nul 2>&1 && set "NEED_VIEWER_DEPS=0"
	) else if "%USE_CONDA_RUN%"=="1" (
		call "%CONDA_CMD%" run -n %CONDA_ENV% python -c "import fastapi, uvicorn" >nul 2>&1 && set "NEED_VIEWER_DEPS=0"
	) else (
		"%PYTHON_CMD%" -c "import fastapi, uvicorn" >nul 2>&1 && set "NEED_VIEWER_DEPS=0"
	)
	if "!NEED_VIEWER_DEPS!"=="1" (
		echo Ensuring Schedule Viewer dependencies are installed...
		if "%USE_CONDA_ACTIVATE%"=="1" (
			call :RunInCondaEnv python -m pip install --disable-pip-version-check -r "%VIEWER_REQUIREMENTS%"
		) else if "%USE_CONDA_RUN%"=="1" (
			call "%CONDA_CMD%" run -n %CONDA_ENV% python -m pip install --disable-pip-version-check -r "%VIEWER_REQUIREMENTS%"
		) else (
			"%PYTHON_CMD%" -m pip install --disable-pip-version-check -r "%VIEWER_REQUIREMENTS%"
		)
		if errorlevel 1 (
			echo.
			echo [ERROR] Failed to install viewer dependencies.
			echo.
			pause
			goto :EOF
		)
	) else (
		echo Viewer dependencies already available. Skipping pip install.
	)
)

echo Starting Schedule Results Viewer (press Ctrl+C to stop)...
echo Open in browser: http://%BROWSER_HOST%:%PORT%

if "%USE_CONDA_ACTIVATE%"=="1" (
	call :RunInCondaEnv python tools\schedule_web_viewer\app.py --results "%RESULTS_DIR%" --host %HOST% --port %PORT%
	set "EXIT_CODE=%ERRORLEVEL%"
) else if "%USE_CONDA_RUN%"=="1" (
	call "%CONDA_CMD%" run -n %CONDA_ENV% python tools\schedule_web_viewer\app.py --results "%RESULTS_DIR%" --host %HOST% --port %PORT%
	set "EXIT_CODE=%ERRORLEVEL%"
) else (
	"%PYTHON_CMD%" tools\schedule_web_viewer\app.py --results "%RESULTS_DIR%" --host %HOST% --port %PORT%
	set "EXIT_CODE=%ERRORLEVEL%"
)
if not "%EXIT_CODE%"=="0" (
	echo.
	echo Viewer exited with code %EXIT_CODE%.
)
echo.
echo Open in browser: http://%BROWSER_HOST%:%PORT%
pause

endlocal
goto :EOF

:RunInCondaEnv
setlocal
if "%CONDA_ACTIVATE_BAT%"=="" (
	echo [ERROR] Unable to locate conda activate script. Set CONDA_ROOT or install Anaconda.
	endlocal & exit /b 1
)
call "%CONDA_ACTIVATE_BAT%" "%CONDA_ENV%" >nul
if errorlevel 1 (
	echo [ERROR] Failed to activate conda environment %CONDA_ENV%.
	endlocal & exit /b 1
)
%*
set "RC=%ERRORLEVEL%"
call conda deactivate >nul 2>&1
endlocal & exit /b %RC%
