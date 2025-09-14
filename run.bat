@echo off
echo Starting Streamlit Article Generator...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run the application
echo.
echo Starting application...
echo Open your browser and go to: http://localhost:8501
echo.
streamlit run app.py

pause
