@echo off
REM ===================================================================
REM  Gunluk otomatik calisma - Windows Gorev Zamanlayici bunu cagirir.
REM
REM  ONEMLI: Bu dosya SAF ASCII olmalidir. Batch yorumlayicisi dosyayi
REM  yerel OEM kod sayfasiyla okur; UTF-8 karakterler (tire, Turkce
REM  harfler) REM satirlarini bozar ve icerik komut olarak calistirilir.
REM
REM  Yaptiklari:
REM    1. Tarama + izleme listesi + ogrenme dongusu
REM    2. Panolari sifreli surume cevirir (DASHBOARD_PASSWORD ayarliysa)
REM    3. Her seyi tarihli bir gunluge yazar, eskileri temizler
REM
REM  Elle de calistirilabilir: cift tikla.
REM ===================================================================

setlocal
cd /d "%~dp0.."

REM --- Gunluk dosyasi. Tarih damgasini Python uretir: Windows'un %DATE%
REM     bicimi yerel ayara gore degisir ve guvenilir degildir.
if not exist "logs" mkdir "logs"
python scripts\_stamp.py > "%TEMP%\hs_stamp.txt" 2>nul
set STAMP=
set /p STAMP=<"%TEMP%\hs_stamp.txt"
if "%STAMP%"=="" set STAMP=bilinmeyen
del "%TEMP%\hs_stamp.txt" 2>nul
set LOG=logs\gunluk_%STAMP%.log

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo BASLANGIC: %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

REM --- 1) Ana dongu: tarama + izleme listesi + ogrenme
python run.py daily --universe smallcap,midcap,wsb --workers 4 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo. >> "%LOG%"
echo daily cikis kodu: %RC% >> "%LOG%"

REM --- 2) Sifreli surum (yalnizca parola ortam degiskeni ayarliysa)
if defined DASHBOARD_PASSWORD (
  echo. >> "%LOG%"
  echo --- sifreli surum uretiliyor --- >> "%LOG%"
  python run.py publish >> "%LOG%" 2>&1
) else (
  echo. >> "%LOG%"
  echo DASHBOARD_PASSWORD ayarli degil, sifreli surum atlandi >> "%LOG%"
)

REM --- 3) GitHub Pages yayini (yalnizca HISSE_REPO ayarliysa)
REM     deploy komutu, sifresiz icerik bulursa gonderimi KENDISI durdurur.
if defined HISSE_REPO (
  if defined DASHBOARD_PASSWORD (
    echo. >> "%LOG%"
    echo --- GitHub Pages yayini --- >> "%LOG%"
    python run.py deploy --repo %HISSE_REPO% >> "%LOG%" 2>&1
  ) else (
    echo. >> "%LOG%"
    echo HISSE_REPO ayarli ama DASHBOARD_PASSWORD yok - yayin ATLANDI >> "%LOG%"
  )
)

REM --- 4) 30 gunden eski gunlukleri sil
forfiles /p logs /m gunluk_*.log /d -30 /c "cmd /c del @path" >nul 2>&1

echo. >> "%LOG%"
echo BITIS: %DATE% %TIME%  (cikis %RC%) >> "%LOG%"
endlocal
exit /b %RC%
