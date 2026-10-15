@echo off
REM Review Lasagna _pts.csv tracks and register the ticked ones to Alyx. Needs ONE-api >= 3.4 (iblenv3.11).
"C:\Users\kimil\anaconda3\envs\iblenv3.11\python.exe" "%~dp0register_tracks_alyx.py" %*
