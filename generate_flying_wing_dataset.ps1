Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$sampleCount = $null
$rootSeed = 20260711
$condaExe = 'C:\Users\zyx20\anaconda3\Scripts\conda.exe'
$generator = Join-Path $repoRoot 'data\generate_flying_wing_dataset.py'
$converter = Join-Path $repoRoot 'data\json_to_typed_obb_dataset.py'
$cache = Join-Path $repoRoot 'data\cst_airfoil_code_cache.pt'
$outputDirPath = Join-Path $repoRoot 'data\flying_wing_dataset'
$datasetPath = Join-Path $outputDirPath 'flying_wing_dataset.pt'

foreach ($path in @($condaExe, $generator, $converter, $cache)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

Push-Location $repoRoot
try {
    $condaHook = & $condaExe 'shell.powershell' 'hook' | Out-String
    Invoke-Expression $condaHook

    conda activate vsppytools
    $sampleCount = [int](& python -c "import sys; sys.path.insert(0, 'data'); import aircraft_dataset_common as common; print(common.DATASET_SAMPLE_COUNT)")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read DATASET_SAMPLE_COUNT from data\aircraft_dataset_common.py"
    }
    $generatorArgs = @('--output-dir', $outputDirPath, '--count', $sampleCount, '--seed', $rootSeed, '--overwrite')
    & python $generator @generatorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Flying-wing OpenVSP/JSON generation failed with exit code $LASTEXITCODE"
    }

    conda activate myml
    & python $converter '--input-dir' $outputDirPath '--output' $datasetPath '--expected-count' $sampleCount
    if ($LASTEXITCODE -ne 0) {
        throw "Flying-wing JSON-to-PT conversion failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
