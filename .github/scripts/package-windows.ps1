param(
    [Parameter(Mandatory = $true)]
    [string] $Version
)

$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') {
    throw "Expected a semantic version, got: $Version"
}
$appVersion = ($Version -split '[-+]')[0]
$target = Join-Path $PSScriptRoot '..\..\nn-approximations\target'
$jar = Join-Path $target "nn-approximations-$Version-all.jar"
if (-not (Test-Path $jar)) {
    throw "Missing release JAR: $jar"
}
$inputDirectory = Join-Path $target 'jpackage-input'
$destination = Join-Path $target 'jpackage-output'
New-Item -ItemType Directory -Path $inputDirectory -Force | Out-Null
Copy-Item $jar (Join-Path $inputDirectory 'nn-approximations.jar')

& jpackage --type app-image --input $inputDirectory --dest $destination `
    --name NeuralNetworkVisualizer --main-jar nn-approximations.jar `
    --main-class net.talaatharb.nn.VisualizationLauncher `
    --app-version $appVersion --vendor 'Talaat Harb' `
    --description 'Neural network training and inference visualizer'
if ($LASTEXITCODE -ne 0) {
    throw "jpackage failed with exit code $LASTEXITCODE"
}

$image = Join-Path $destination 'NeuralNetworkVisualizer'
Copy-Item (Join-Path $PSScriptRoot '..\..\LICENSE') (Join-Path $image 'LICENSE.txt')
Copy-Item (Join-Path $PSScriptRoot '..\..\README.md') (Join-Path $image 'README.md')
$executable = (Resolve-Path (Join-Path $image 'NeuralNetworkVisualizer.exe')).Path
# Launch the packaged image, not the Maven classpath, to catch missing runtime/native dependencies.
$stderr = Join-Path $target 'portable-stderr.log'
$process = Start-Process -FilePath $executable -PassThru `
    -RedirectStandardOutput (Join-Path $target 'portable-stdout.log') -RedirectStandardError $stderr
$processIds = [System.Collections.Generic.HashSet[int]]::new()
$null = $processIds.Add($process.Id)
try {
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Milliseconds 500
        # jpackage can spawn a child launcher; the window belongs to that child, not the bootstrap process.
        Get-CimInstance Win32_Process | Where-Object {
            $_.ExecutablePath -eq $executable -and $processIds.Contains([int]$_.ParentProcessId)
        } | ForEach-Object { $null = $processIds.Add([int]$_.ProcessId) }
        $running = @($processIds | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        $window = $running | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Responding }
        if ($running.Count -eq 0) {
            throw "Portable application exited during startup. $(Get-Content $stderr -Raw)"
        }
    } while (-not $window -and (Get-Date) -lt $deadline)
    if (-not $window) {
        throw "Portable application did not create a responsive window. $(Get-Content $stderr -Raw)"
    }
} finally {
    foreach ($id in $processIds) {
        $running = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($running) {
            Stop-Process -Id $id
            $running.WaitForExit()
        }
    }
}

$archive = Join-Path $target "nn-approximations-$Version-Windows-x64-portable.zip"
Compress-Archive -Path $image -DestinationPath $archive
