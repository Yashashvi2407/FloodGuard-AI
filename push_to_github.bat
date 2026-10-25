@echo off
echo ============================================================
echo   Uploading FloodGuard AI to GitHub
echo ============================================================
echo.
cd /d "C:\Users\Yashashvi\.gemini\antigravity\scratch\FloodGuard_Project"
git remote remove origin 2>nul
git remote add origin https://github.com/Yashashvi2407/FloodGuard-AI.git
git branch -M main
echo Pushing your files to https://github.com/Yashashvi2407/FloodGuard-AI ...
echo (If a browser window pops up, click 'Sign in with your browser')
echo.
git push -f -u origin main
echo.
echo ============================================================
echo   Done! All files are now in your GitHub repository!
echo ============================================================
pause
