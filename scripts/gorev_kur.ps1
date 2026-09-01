<#
    Gunluk otomasyonu Windows Gorev Zamanlayici'ya kurar.

    NEDEN XML
    ---------
    Register-ScheduledTask bu makinede "Erisim engellendi" (0x80070005)
    veriyor; ilk kurulumun calisan tek yolu tanimi XML olarak uretip
    schtasks.exe ile kaydetmek. (Gorev BIR KEZ kurulduktan sonra
    Set-ScheduledTask ile degistirilebiliyor -- 27.08.2026'da tetikler
    boyle degistirildi. Kisitlama yalnizca yeni kayitta.)

    XML UTF-16 (Unicode) olmak ZORUNDA. UTF-8 yazilirsa schtasks dosyayi
    reddeder. Yol icinde Turkce karakter oldugu icin bu ayrica onemli.

    TASARIM
    -------
    * GUNDE TEK TETIK: 07:00, tekrar yok.
    * StartWhenAvailable: bilgisayar 07:00'da kapaliysa acilir acilmaz
      telafi calismasi yapilir. Ayri bir LogonTrigger'a gerek YOK.
    * IgnoreNew: onceki calisma surerken yenisi baslatilmaz.

    27.08.2026 -- ONCEKI TASARIM VE NEDEN BIRAKILDI
    Once "07:00'dan itibaren 2 saatte bir, 14 saat" + oturum acilisi
    tetigi vardi: gunde 9 deneme. Gun kilidi (logs\son_basari.txt)
    tekrarlarin IS yapmasini engelliyordu ama her tetik yine de bir konsol
    penceresi aciyordu. Daha kotusu, kullanici o pencereyi kapatinca
    tarama yarida kaliyor, gun kilidi dusmuyor ve sonraki tetik ayni isi
    bastan basliyordu -- kendini besleyen bir dongu. Tek tetik +
    StartWhenAvailable ayni telafi garantisini pencere yagmuru olmadan
    veriyor.

    Kullanim:  powershell -ExecutionPolicy Bypass -File scripts\gorev_kur.ps1
               powershell ... -File scripts\gorev_kur.ps1 -Kaldir
#>
param(
    [string]$GorevAdi = 'HisseSiralama_Gunluk',
    [string]$Saat     = '07:00:00',
    [switch]$Kaldir
)

$ErrorActionPreference = 'Stop'
$kok = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $kok 'scripts\gunluk.bat'

if ($Kaldir) {
    schtasks /Delete /TN $GorevAdi /F
    Write-Host "Gorev kaldirildi: $GorevAdi"
    exit 0
}

if (-not (Test-Path $bat)) { throw "gunluk.bat bulunamadi: $bat" }

$kullanici = "$env:USERDOMAIN\$env:USERNAME"
$bugun     = (Get-Date).ToString('yyyy-MM-dd')

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Hisse siralama motoru - gunluk tarama, izleme listesi, ogrenme dongusu, yayin.</Description>
    <URI>\$GorevAdi</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>${bugun}T$Saat</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$kullanici</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT15M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$bat</Command>
      <WorkingDirectory>$kok</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

$gecici = Join-Path $env:TEMP 'hisse_gorev.xml'
# UTF-16 SART: schtasks UTF-8 XML kabul etmiyor.
$xml | Out-File -FilePath $gecici -Encoding Unicode -Force

schtasks /Create /TN $GorevAdi /XML $gecici /F
Remove-Item $gecici -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Kuruldu: $GorevAdi"
Write-Host "  komut     : $bat"
Write-Host "  tetik     : gunde 1 kez ($Saat)"
Write-Host "  telafi    : bilgisayar o saatte kapaliysa ilk acilista calisir"
Write-Host "  gun kilidi: ayni gun ikinci calisma is yapmaz (logs\son_basari.txt)"
Write-Host ""
Write-Host "Durum icin : python scripts\durum.py"
Write-Host "Elle calis : schtasks /Run /TN $GorevAdi"
Write-Host "Kaldirmak  : powershell -File scripts\gorev_kur.ps1 -Kaldir"
