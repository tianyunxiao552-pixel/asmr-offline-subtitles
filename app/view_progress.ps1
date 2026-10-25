$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$toolRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $toolRoot '.logs'
$script:mainLog = $null
$script:lastText = ''
$script:tick = 0

function Read-SharedLog([string]$Path, [bool]$Head = $false) {
    $sharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $sharing)
    try {
        if ($Head) {
            $buffer = New-Object byte[] 4096
            $length = $stream.Read($buffer, 0, $buffer.Length)
            return [Text.Encoding]::UTF8.GetString($buffer, 0, $length)
        }
        if ($stream.Length -gt 131072) { [void]$stream.Seek(-131072, [IO.SeekOrigin]::End) }
        $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}

$form = New-Object Windows.Forms.Form
$form.Text = 'ASMR 实时进度'
$form.Size = New-Object Drawing.Size(1120, 740)
$form.MinimumSize = New-Object Drawing.Size(740, 460)
$form.StartPosition = 'CenterScreen'
$form.BackColor = [Drawing.Color]::FromArgb(24, 26, 30)
$header = New-Object Windows.Forms.Label
$header.Dock = 'Top'
$header.Height = 74
$header.Padding = New-Object Windows.Forms.Padding(14, 10, 10, 5)
$header.Font = New-Object Drawing.Font('Microsoft YaHei UI', 11)
$header.ForeColor = [Drawing.Color]::White
$header.Text = '正在连接本地处理日志……'
$hint = New-Object Windows.Forms.Label
$hint.Dock = 'Bottom'
$hint.Height = 34
$hint.Padding = New-Object Windows.Forms.Padding(14, 5, 0, 0)
$hint.ForeColor = [Drawing.Color]::LightGray
$hint.Text = '每秒刷新｜本窗口仅查看日志；关闭它不影响 run.bat。恢复处理请双击 run.bat。'
$logView = New-Object Windows.Forms.RichTextBox
$logView.Dock = 'Fill'
$logView.ReadOnly = $true
$logView.BackColor = [Drawing.Color]::FromArgb(16, 18, 22)
$logView.ForeColor = [Drawing.Color]::FromArgb(210, 225, 215)
$logView.Font = New-Object Drawing.Font('Microsoft YaHei UI', 10)
$logView.WordWrap = $false
$logView.BorderStyle = 'None'
$form.Controls.Add($logView)
$form.Controls.Add($header)
$form.Controls.Add($hint)
$timer = New-Object Windows.Forms.Timer
$timer.Interval = 1000
$timer.Add_Tick({
    try {
        if (($script:tick % 5) -eq 0 -or !$script:mainLog) {
            $candidates = Get-ChildItem -LiteralPath $logRoot -Filter '20*.log' -File -ErrorAction SilentlyContinue |
                Where-Object { !$script:mainLog -or $_.Name -gt $script:mainLog.Name } |
                Sort-Object Name -Descending
            foreach ($candidate in $candidates) {
                $head = Read-SharedLog $candidate.FullName $true
                if ($head -match 'GPU 0:|质量模式：|固定分离 |没有待处理 WAV|首次运行先准备|目录整理完成') {
                    $script:mainLog = $candidate
                    break
                }
            }
        }
        $script:tick++
        if (!$script:mainLog) { $header.Text = '等待处理日志。若尚未运行，请双击 run.bat。'; return }
        $text = Read-SharedLog $script:mainLog.FullName
        if ($text -ne $script:lastText) {
            $logView.Text = $text.Replace("`n", "`r`n").Replace("`r`r`n", "`r`n")
            $logView.SelectionStart = $logView.TextLength
            $logView.ScrollToCaret()
            $script:lastText = $text
        }
        $files = @(Get-ChildItem -LiteralPath (Join-Path $toolRoot '待翻译') -Filter '*.wav' -File -Recurse -ErrorAction SilentlyContinue)
        $finished = @($files | Where-Object { Test-Path -LiteralPath ([IO.Path]::ChangeExtension($_.FullName, '.srt')) }).Count
        $running = $false
        $lockPath = Join-Path $toolRoot '.tool.lock'
        if (Test-Path -LiteralPath $lockPath) {
            try {
                $check = [IO.File]::Open($lockPath, 'Open', 'Read', 'None')
                $check.Dispose()
            } catch [IO.IOException] { $running = $true }
        }
        $status = if ($running) { '工具正在运行' } else { '工具未运行；以下为上次日志' }
        $header.Text = $status + '    ' + ('待翻译目录：{0} / {1} 条已生成中文字幕    刷新：{2}' -f $finished, $files.Count, (Get-Date -Format 'HH:mm:ss')) +
                       "`r`n当前日志：" + $script:mainLog.Name + '    下方显示实际文件、阶段与字幕句数。'
    } catch {
        $header.Text = '暂时无法读取日志，正在自动重试：' + $_.Exception.Message
    }
})
$form.Add_Shown({ $timer.Start() })
$form.Add_FormClosed({ $timer.Stop(); $timer.Dispose() })
[void]$form.ShowDialog()
