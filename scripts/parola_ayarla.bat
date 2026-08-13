@echo off
REM Parola ayarlama yardimcisi. Bu pencerede yazdiklarin sohbet kaydina gecmez.
title Parola Ayarlama
color 0B
echo.
echo  ============================================================
echo   PANO PAROLASI AYARLAMA
echo  ============================================================
echo.
echo   Buraya yazdigin parola SOHBET KAYDINA GECMEZ.
echo.
echo   Guclu parola: en az 5 rastgele kelime veya 16+ karakter.
echo   Ornek bicim : masa-bulut-kirmizi-tren-ceviz
echo.
set /p PW=Parolayi yaz ve Enter'a bas: 
if "%PW%"=="" (
  echo.
  echo  Bos parola kabul edilmez. Pencereyi kapatip tekrar dene.
  pause
  exit /b 1
)
setx DASHBOARD_PASSWORD "%PW%" >nul
setx HISSE_REPO "yigaykut/hisse-pano" >nul
set PW=
echo.
echo  ============================================================
echo   TAMAM. Parola ve depo adi kaydedildi.
echo   Bu pencereyi kapatip Claude Code'a "tamam" yaz.
echo  ============================================================
echo.
pause
