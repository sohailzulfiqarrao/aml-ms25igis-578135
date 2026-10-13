@echo off
cd /d "%~dp0"
git add -A
git commit -m "Updating coursework"
git push
pause
