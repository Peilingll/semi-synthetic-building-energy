@echo off
REM ---------------------------------------------------------------------------
REM Build thesis.pdf with the portable MiKTeX installation on D:.
REM
REM   build.cmd          full build (pdflatex -> biber -> glossaries -> x2)
REM   build.cmd quick    single pdflatex pass, for text-only edits
REM   build.cmd watch    rebuild automatically on every file save (Ctrl+C stops)
REM   build.cmd clean    delete auxiliary files
REM
REM Run a full build whenever citations, cross-references, the table of
REM contents, or the list of abbreviations changed.
REM ---------------------------------------------------------------------------

set "MIKTEX=D:\MiKTeX\texmfs\install\miktex\bin\x64"
set "PATH=%MIKTEX%;%PATH%"
set "MAIN=thesis"
set "PDFLATEX=%MIKTEX%\pdflatex.exe -interaction=nonstopmode -synctex=1 --enable-installer"

cd /d "%~dp0"

if /i "%~1"=="clean" goto :clean
if /i "%~1"=="quick" goto :quick
if /i "%~1"=="watch" goto :watch

echo [1/5] pdflatex
%PDFLATEX% %MAIN%.tex >nul
echo [2/5] biber
"%MIKTEX%\biber.exe" %MAIN%
echo [3/5] makeglossaries
"%MIKTEX%\makeglossaries-lite.exe" %MAIN% >nul
echo [4/5] pdflatex
%PDFLATEX% %MAIN%.tex >nul
echo [5/5] pdflatex
%PDFLATEX% %MAIN%.tex >nul
goto :report

:quick
echo [1/1] pdflatex
%PDFLATEX% %MAIN%.tex >nul
goto :report

:watch
echo Watching for changes - press Ctrl+C to stop.
"%MIKTEX%\latexmk.exe" -pdf -pvc -interaction=nonstopmode -synctex=1 %MAIN%.tex
goto :eof

:clean
del /q %MAIN%.aux %MAIN%.bbl %MAIN%.bcf %MAIN%.blg %MAIN%.glg* %MAIN%.glo* ^
       %MAIN%.gls* %MAIN%.glsdefs %MAIN%.ist %MAIN%.lof %MAIN%.log %MAIN%.lot ^
       %MAIN%.out %MAIN%.run.xml %MAIN%.synctex.gz %MAIN%.toc 2>nul
del /q content\*.aux 2>nul
echo Auxiliary files removed.
goto :eof

:report
echo.
findstr /c:"Output written" %MAIN%.log
echo.
echo --- LaTeX errors ---
findstr /b /c:"!" %MAIN%.log || echo none
echo.
echo --- unresolved references / citations ---
findstr /c:"undefined" %MAIN%.log || echo none
echo.
echo --- content wider than the text block ---
findstr /c:"Overfull \hbox" %MAIN%.log || echo none
