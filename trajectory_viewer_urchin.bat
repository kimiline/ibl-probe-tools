@echo off
REM Needs Python >= 3.10 with ONE-api >= 3.4 (iblenv3.11); older ONE cannot query Alyx.
C:\Users\kimil\anaconda3\envs\iblenv3.11\python.exe "%~dp0trajectory_viewer_urchin.py" %*
