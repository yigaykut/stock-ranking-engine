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
REM    3. GitHub Pages yayini (HISSE_REPO ayarliysa)
REM    4. Her seyi tarihli bir gunluge yazar, eskileri temizler
REM
REM  GUNDE BIR KEZ kurali dosyayla saglanir: basarili bir calisma
REM  logs\son_basari.txt icine gunun tarihini yazar. Ayni gun tekrar
REM  cagrilirsa is yapmadan cikar. Bu sayede gorev zamanlayiciya birden
REM  cok tetik (07:00, oturum acilisi, gun icinde periyodik) koyabiliriz;
REM  bilgisayar sabah kapaliysa ilk acilista telafi calismasi yapilir.
REM
REM  Zorla calistirmak icin:  gunluk.bat force
REM ===================================================================

setlocal
cd /d "%~dp0.."

REM --- Python cikisi UTF-8 olsun. Varsayilan konsol kod sayfasi cp1254 ve
REM     ASCII disi TEK bir karakter iceren print, tum sureci
REM     UnicodeEncodeError ile dusuruyordu. run.py bunu kendi icinde de
REM     sabitliyor; bu satir alt sureclerde (publish/deploy) ayni garantiyi
REM     verir.
set PYTHONIOENCODING=utf-8

REM --- Tarih damgasini Python uretir: Windows'un %DATE% bicimi yerel
REM     ayara gore degisir ve guvenilir degildir.
if not exist "logs" mkdir "logs"
python scripts\_stamp.py > "%TEMP%\hs_stamp.txt" 2>nul
set STAMP=
set /p STAMP=<"%TEMP%\hs_stamp.txt"
if "%STAMP%"=="" set STAMP=bilinmeyen
del "%TEMP%\hs_stamp.txt" 2>nul
set LOG=logs\gunluk_%STAMP%.log
set MARK=logs\son_basari.txt

REM --- Bugun zaten basariyla calistiysa cik (force ile atlanir)
if /i "%~1"=="force" goto :calis
if not exist "%MARK%" goto :calis
set DONE=
set /p DONE=<"%MARK%"
if "%DONE%"=="%STAMP%" (
  echo [%TIME%] %STAMP% bugun zaten tamamlandi - atlandi. >> "%LOG%"
  endlocal
  exit /b 0
)

:calis
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
REM     NOT: parantezli bloklarda %ERRORLEVEL% blok AYRISTIRILIRKEN
REM     genisletilir, yani komuttan onceki degeri okur. Bu yuzden cikis
REM     kodu yakalanan yerlerde parantez degil goto kullaniliyor.
set RCP=0
if not defined DASHBOARD_PASSWORD goto :yayin_yok
echo. >> "%LOG%"
echo --- sifreli surum uretiliyor --- >> "%LOG%"
python run.py publish >> "%LOG%" 2>&1
set RCP=%ERRORLEVEL%
echo sifreleme cikis kodu: %RCP% >> "%LOG%"
goto :sifre_bitti
:yayin_yok
echo. >> "%LOG%"
echo DASHBOARD_PASSWORD ayarli degil, sifreli surum atlandi >> "%LOG%"
:sifre_bitti

REM --- 3) GitHub Pages yayini (yalnizca HISSE_REPO ayarliysa)
REM     deploy komutu, sifresiz icerik bulursa gonderimi KENDISI durdurur.
set RCD=0
if not defined HISSE_REPO goto :dagit_bitti
if not defined DASHBOARD_PASSWORD goto :dagit_parolasiz
echo. >> "%LOG%"
echo --- GitHub Pages yayini --- >> "%LOG%"
python run.py deploy --repo %HISSE_REPO% >> "%LOG%" 2>&1
set RCD=%ERRORLEVEL%
echo yayin cikis kodu: %RCD% >> "%LOG%"
goto :dagit_bitti
:dagit_parolasiz
echo. >> "%LOG%"
echo HISSE_REPO ayarli ama DASHBOARD_PASSWORD yok - yayin ATLANDI >> "%LOG%"
:dagit_bitti

REM --- 4) Tum adimlar temiz bittiyse gunu isaretle
REM     NOT: "if A if B if C (...) else (...)" yazilamaz - else yalnizca EN
REM     ICTEKI if'e baglanir, disaridaki kosul patlarsa hicbir dal calismaz ve
REM     hata sessizce loga dusmez. Bu yuzden tek bir bayrak kullaniliyor.
set OK=1
if not "%RC%"=="0" set OK=0
if not "%RCP%"=="0" set OK=0
if not "%RCD%"=="0" set OK=0

if "%OK%"=="1" goto :isaretle
echo. >> "%LOG%"
echo HATA VAR - gun isaretlenmedi, sonraki tetik tekrar deneyecek >> "%LOG%"
echo   daily=%RC%  publish=%RCP%  deploy=%RCD% >> "%LOG%"
goto :temizlik
:isaretle
> "%MARK%" echo %STAMP%
echo. >> "%LOG%"
echo gun isaretlendi: %STAMP% >> "%LOG%"
:temizlik

REM --- 5) 30 gunden eski gunlukleri sil
forfiles /p logs /m gunluk_*.log /d -30 /c "cmd /c del @path" >nul 2>&1

echo. >> "%LOG%"
echo BITIS: %DATE% %TIME%  (daily=%RC% publish=%RCP% deploy=%RCD%) >> "%LOG%"
endlocal
exit /b %RC%
