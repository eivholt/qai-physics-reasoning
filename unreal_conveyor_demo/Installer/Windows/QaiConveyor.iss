#ifndef GameSource
  #error GameSource must point to the archived Windows Unreal build
#endif
#ifndef ProvisionerExe
  #error ProvisionerExe must point to qai-conveyor-setup.exe
#endif
#ifndef PayloadSource
  #error PayloadSource must point to the prepared private release payload
#endif
#ifndef OutputDirectory
  #define OutputDirectory "."
#endif

[Setup]
AppId=QaiConveyorDemo.0.1
AppName=Reason2 Conveyor Safety
AppVersion=0.1.0
AppPublisher=Private physical-AI demo
DefaultDirName={localappdata}\Programs\QaiConveyorDemo
DefaultGroupName=Reason2 Conveyor Safety
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir={#OutputDirectory}
OutputBaseFilename=Reason2-Conveyor-Safety-Windows-0.1.0
SetupLogging=yes
UninstallLogging=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Files]
Source: "{#GameSource}\*"; DestDir: "{app}\Game"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProvisionerExe}"; DestDir: "{app}\Provisioner"; DestName: "qai-conveyor-setup.exe"; Flags: ignoreversion
Source: "{#PayloadSource}\*"; DestDir: "{app}\Payload"; Excludes: "EVK\*.tar.gz"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadSource}\EVK\evk-geniex-runtime-v0317-qairt245.tar.gz"; DestDir: "{app}\Payload\EVK"; Flags: ignoreversion nocompression
Source: "{#PayloadSource}\EVK\cosmos-reason2-parcel-speed-v1-cl512-256x448-w8text-geniex-qairt245-os19-r1.tar.gz"; DestDir: "{app}\Payload\EVK"; Flags: ignoreversion nocompression

[Icons]
Name: "{group}\Reason2 Conveyor Safety"; Filename: "{app}\Provisioner\qai-conveyor-setup.exe"; Parameters: "launch --app-dir ""{app}"""; WorkingDir: "{app}"
Name: "{group}\Repair models and EVK"; Filename: "{app}\Provisioner\qai-conveyor-setup.exe"; Parameters: "ensure --app-dir ""{app}"""; WorkingDir: "{app}"
Name: "{group}\Collect support bundle"; Filename: "{app}\Provisioner\qai-conveyor-setup.exe"; Parameters: "support-bundle --app-dir ""{app}"" --reason ""Start Menu request"""; WorkingDir: "{app}"
Name: "{autodesktop}\Reason2 Conveyor Safety"; Filename: "{app}\Provisioner\qai-conveyor-setup.exe"; Parameters: "launch --app-dir ""{app}"""; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Code]
var
  ProvisionExitCode: Integer;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if CompareText(ExpandConstant('{param:SKIPPROVISION|0}'), '1') = 0 then
    begin
      Log('Skipping provisioning because /SKIPPROVISION=1 was supplied for installer verification');
      Exit;
    end;
    Log('Starting logged host/EVK provisioning');
    if not Exec(
      ExpandConstant('{app}\Provisioner\qai-conveyor-setup.exe'),
      'install --app-dir "' + ExpandConstant('{app}') + '"',
      ExpandConstant('{app}'),
      SW_SHOW,
      ewWaitUntilTerminated,
      ProvisionExitCode
    ) then
      RaiseException('Could not start the Qai Conveyor provisioner.');
    Log('Provisioner exit code: ' + IntToStr(ProvisionExitCode));
    if ProvisionExitCode <> 0 then
      RaiseException(
        'Model/EVK provisioning failed. A redacted support ZIP was created in ' +
        ExpandConstant('{localappdata}\QaiConveyorDemo\Logs')
      );
  end;
end;

procedure DeinitializeSetup;
var
  Destination: String;
begin
  ForceDirectories(ExpandConstant('{localappdata}\QaiConveyorDemo\Logs'));
  Destination := ExpandConstant('{localappdata}\QaiConveyorDemo\Logs\inno-setup-') +
    GetDateTimeString('yyyymmdd-hhnnss', '-', ':') + '.log';
  if FileExists(ExpandConstant('{log}')) then
    CopyFile(ExpandConstant('{log}'), Destination, False);
end;
