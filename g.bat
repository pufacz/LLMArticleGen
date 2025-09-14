@echo off
SETLOCAL

:: Check if a commit message was provided
IF "%~1"=="" (
    ECHO Usage: %~nx0 "Your commit message here"
    ECHO Example: %~nx0 "Feat: Added new login screen"
    ECHO.
    ECHO Please provide a commit message as the first argument.
    EXIT /B 1
)

:: Store the commit message from the first argument
:: %~1 removes any surrounding quotes if the user provides them
SET "COMMIT_MESSAGE=%~1"

ECHO.
ECHO --- Git Push Automation (Windows) ---
ECHO Commit Message: "%COMMIT_MESSAGE%"
ECHO.

:: Step 1: git add .
ECHO Running: git add .
git add .
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error during git add. Aborting.
    GOTO :END
)

:: Step 2: git commit -m "commit_message"
ECHO Running: git commit -m "%COMMIT_MESSAGE%"
git commit -m "%COMMIT_MESSAGE%"
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error during git commit. This might be because there are no changes to commit, or a merge conflict.
    GOTO :END
)

:: Step 3: git push
ECHO Running: git push
git push
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error during git push. Check your network, credentials, or remote status.
    GOTO :END
)

ECHO.
ECHO --- Git Push Successful! ---

:END
ENDLOCAL
