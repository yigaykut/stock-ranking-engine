' ===================================================================
'  gunluk.bat'i PENCERESIZ calistirir.
'
'  Neden var: gorev zamanlayici gunluk.bat'i dogrudan cagirdiginda
'  ekranda 30-40 dakika duran bir cmd penceresi aciliyordu. Pencere
'  kapatilinca tarama yarida kaliyor, cikis kodu 3221225786
'  (STATUS_CONTROL_C_EXIT) donuyor ve gun isaretlenmiyor -- yani
'  sistem gunlerce guncellenmiyor. 03.09 ve 04.09 taramalari tam
'  olarak boyle olmustu.
'
'  Bu dosya SAF ASCII olmalidir; wscript dosyayi yerel kod sayfasiyla
'  okur ve Turkce karakterler yolu bozar.
'
'  Kullanim:  wscript.exe //B gunluk_sessiz.vbs [force]
' ===================================================================
Option Explicit

Dim fso, sh, klasor, bat, arg, komut, kod

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

klasor = fso.GetParentFolderName(WScript.ScriptFullName)
bat    = fso.BuildPath(klasor, "gunluk.bat")

If Not fso.FileExists(bat) Then
  WScript.Quit 2
End If

arg = ""
If WScript.Arguments.Count > 0 Then
  arg = " " & WScript.Arguments(0)
End If

komut = "cmd.exe /c """"" & bat & """" & arg & """"

' 0 = pencere gizli, True = bitene kadar bekle.
' Beklemek sart: gorev zamanlayici cikis kodunu bu surecten okuyor.
kod = sh.Run(komut, 0, True)
WScript.Quit kod
