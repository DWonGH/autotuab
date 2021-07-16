@echo off

set SRCDIR=H:\TUAB_relabelled
set DESTDIR=H:\TUAB2_relabelled

REM Create new version of TUAB to match the format expected by auto-eeg-diagnosis-example code.
xcopy %SRCDIR%\v2.0.0\edf\eval\abnormal %DESTDIR%\abnormal\edf\eval\ /E/H
xcopy %SRCDIR%\v2.0.0\edf\train\abnormal %DESTDIR%\abnormal\edf\train\ /E/H
xcopy %SRCDIR%\v2.0.0\edf\eval\normal %DESTDIR%\normal\edf\eval\ /E/H
xcopy %SRCDIR%\v2.0.0\edf\train\normal %DESTDIR%\normal\edf\train\ /E/H