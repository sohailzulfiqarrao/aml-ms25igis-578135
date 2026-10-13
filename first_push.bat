@echo off
cd /d "%~dp0"
git init
git add -A
git commit -m "Applied Machine Learning: notebooks, assignment and landing page"
git branch -M main
git remote add origin https://github.com/sohailzulfiqarrao/aml-ms25igis-578135
git push -u origin main
pause
