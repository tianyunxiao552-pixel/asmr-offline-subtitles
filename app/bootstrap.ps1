param(
    [ValidateSet('run', 'clean')][string]$Mode = 'run',
    [switch]$Offline,
    [switch]$Prepare,
    [switch]$Diagnose,
    [switch]$SelfTest,
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$toolRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$runtimeDir = Join-Path $toolRoot '.runtime'
$modelDir = Join-Path $toolRoot '.models'
$lockStream = $null
$exitCode = 0

function Assert-ToolRoot {
    $marker = Join-Path $toolRoot '.asmr-tool-id'
    if (!(Test-Path -LiteralPath $marker -PathType Leaf) -or
        ([IO.File]::ReadAllText($marker).Trim() -ne 'asmr-offline-subtitles-v1-683f8b0e')) {
        throw 'Tool marker missing. Keep the complete tool folder together.'
    }
    # Refuse junctions/symlinks in every ancestor, including the tool directory.
    $item = Get-Item -LiteralPath $toolRoot -Force
    while ($null -ne $item) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw ('Junction/symlink is not supported: ' + $item.FullName)
        }
        $item = $item.Parent
    }
}

function Assert-CacheTree {
    $resolved = [IO.Path]::GetFullPath($modelDir)
    $expected = $toolRoot.TrimEnd('\') + '\.models'
    if ($resolved -ne $expected -or $resolved -eq [IO.Path]::GetPathRoot($resolved)) {
        throw ('Unsafe cache target: ' + $resolved)
    }
    if (Test-Path -LiteralPath $resolved) {
        $rootItem = Get-Item -LiteralPath $resolved -Force
        if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw ('Refusing cache junction/symlink: ' + $resolved)
        }
        # Walk one directory at a time: never recurse through a reparse point.
        $pending = New-Object 'System.Collections.Generic.Stack[string]'
        $pending.Push($resolved)
        while ($pending.Count -gt 0) {
            $folder = $pending.Pop()
            foreach ($entry in Get-ChildItem -LiteralPath $folder -Force) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw ('Refusing cache junction/symlink: ' + $entry.FullName)
                }
                if ($entry.PSIsContainer) { $pending.Push($entry.FullName) }
            }
        }
    }
}

function Fetch-File([string]$Url, [string]$Target) {
    if ($Offline) { throw ('Offline mode: missing ' + $Target) }
    $parentDir = Split-Path -Parent $Target
    [IO.Directory]::CreateDirectory($parentDir) | Out-Null
    Write-Host ('Downloading: ' + $Url)
    $partial = $Target + '.part'
    # curl shows progress; failed bootstrap downloads are retried on the next run.
    $curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curlCmd) {
        & $curlCmd.Source --fail --location --retry 3 --connect-timeout 30 --output $partial $Url
        if ($LASTEXITCODE -ne 0) { throw ('Download failed: ' + $Url) }
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
    }
    Move-Item -LiteralPath $partial -Destination $Target -Force
}

function Run-Python([string[]]$Arguments) {
    & $script:pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Python command failed (exit ' + $LASTEXITCODE + '). See messages above.') }
}

try {
    Assert-ToolRoot
    try {
        $lockStream = [IO.File]::Open((Join-Path $toolRoot '.tool.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { throw 'This tool is already running or cleaning. Close that run first.' }
    if ($Mode -eq 'clean') {
        Assert-CacheTree
        if (Test-Path -LiteralPath $modelDir) {
            $bytes = (Get-ChildItem -LiteralPath $modelDir -Recurse -File -Force | Measure-Object -Property Length -Sum).Sum
            Write-Host ('Removing only this tool model cache: ' + $modelDir)
            Remove-Item -LiteralPath $modelDir -Recurse -Force
            Write-Host ('Done. Freed approximately {0:N2} GiB. Audio, subtitles and runtime were kept.' -f ($bytes / 1GB))
        } else { Write-Host 'No downloaded model cache to clean.' }
    } else {
        $inboxName = -join [char[]]@(0x5F85, 0x7FFB, 0x8BD1)
        $outboxName = -join [char[]]@(0x5DF2, 0x7FFB, 0x8BD1)
        $inboxDir = Join-Path $toolRoot $inboxName
        $outboxDir = Join-Path $toolRoot $outboxName
        foreach ($queueDir in @($inboxDir, $outboxDir)) {
            if ((Test-Path -LiteralPath $queueDir) -and
                (((Get-Item -LiteralPath $queueDir -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
                throw ('Queue directory cannot be a junction/symlink: ' + $queueDir)
            }
            [IO.Directory]::CreateDirectory($queueDir) | Out-Null
        }
        $wavFiles = @(Get-ChildItem -LiteralPath $toolRoot -File | Where-Object { $_.Extension -ieq '.wav' })
        if (!$Prepare -and !$Diagnose -and !$SelfTest -and !$Rebuild) {
            $incomingEntries = @(Get-ChildItem -LiteralPath $inboxDir -Force)
            if ($wavFiles.Count -eq 0 -and $incomingEntries.Count -eq 0) {
                Write-Host ('Drop entire work folders into: ' + $inboxDir)
                Write-Host ('Completed work folders will be moved to: ' + $outboxDir)
                return
            }
        }
        [IO.Directory]::CreateDirectory($runtimeDir) | Out-Null
        Assert-CacheTree
        $script:pythonExe = Join-Path $runtimeDir 'python\python.exe'
        $pythonDir = Split-Path -Parent $script:pythonExe
        $pythonReady = Join-Path $pythonDir '.python-ready'
        if (!(Test-Path -LiteralPath $pythonReady)) {
            $zip = Join-Path $runtimeDir 'python.zip'
            Fetch-File 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' $zip
            Expand-Archive -LiteralPath $zip -DestinationPath $pythonDir -Force
            Remove-Item -LiteralPath $zip -Force
            # Embedded Python is isolated from system Python and user packages.
            [IO.File]::WriteAllText((Join-Path $pythonDir 'python312._pth'), "python312.zip`r`n.`r`nLib\site-packages`r`nimport site`r`n")
            [IO.File]::WriteAllText($pythonReady, '3.12.10')
        }
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $env:PYTHONNOUSERSITE = '1'
        $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
        $env:PIP_NO_CACHE_DIR = '1'
        $env:HF_HUB_DISABLE_TELEMETRY = '1'
        $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
        $env:HF_HOME = Join-Path $modelDir 'huggingface'
        $env:TORCH_HOME = Join-Path $modelDir 'torch'
        $env:XDG_CACHE_HOME = Join-Path $modelDir 'other-cache'
        $reqPath = Join-Path $PSScriptRoot 'requirements.txt'
        # Use .NET directly: some Windows PowerShell environments do not expose
        # the Get-FileHash cmdlet through module auto-loading.
        $hashAlgorithm = [Security.Cryptography.SHA256]::Create()
        try {
            $requirementsHash = [BitConverter]::ToString($hashAlgorithm.ComputeHash([IO.File]::ReadAllBytes($reqPath))).Replace('-', '')
        } finally { $hashAlgorithm.Dispose() }
        $fingerprint = $requirementsHash + '-torch280-cu128-quality-v2'
        $readyFile = Join-Path $runtimeDir 'dependencies.ready'
        $depsReady = (Test-Path -LiteralPath $readyFile) -and ([IO.File]::ReadAllText($readyFile) -eq $fingerprint)
        if (!$SelfTest -and !$depsReady) {
            if ($Offline) { throw 'Dependencies are not installed. Run once online first.' }
            $getPip = Join-Path $runtimeDir 'get-pip.py'
            Fetch-File 'https://bootstrap.pypa.io/get-pip.py' $getPip
            Run-Python @($getPip, '--no-warn-script-location')
            Remove-Item -LiteralPath $getPip -Force
            Run-Python @('-m', 'pip', 'install', '--only-binary=:all:', 'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
            Run-Python @('-m', 'pip', 'install', '--prefer-binary', '-r', $reqPath)
            Run-Python @('-m', 'pip', 'install', '--no-deps', 'faster-whisper==1.2.1')
            Run-Python @((Join-Path $PSScriptRoot 'pipeline.py'), 'check-deps')
            [IO.File]::WriteAllText($readyFile, $fingerprint)
        }
        $argsList = @((Join-Path $PSScriptRoot 'pipeline.py'))
        if ($SelfTest) { $argsList += 'self-test' }
        elseif ($Diagnose) { $argsList += 'diagnose' }
        elseif ($Prepare) { $argsList += 'prepare' }
        elseif ($Rebuild) { $argsList += 'rebuild' }
        else { $argsList += 'run' }
        if ($Offline) { $argsList += '--offline' }
        Run-Python $argsList
    }
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
}
exit $exitCode
