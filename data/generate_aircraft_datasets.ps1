Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$condaExe = 'C:\Users\zyx20\anaconda3\Scripts\conda.exe'
$mymlPython = 'C:\Users\zyx20\anaconda3\envs\myml\python.exe'
$flyingWingGenerator = Join-Path $PSScriptRoot 'generate_flying_wing_dataset.py'
$conventionalCanardGenerator = Join-Path $PSScriptRoot 'generate_conventional_canard_dataset.py'
$converter = Join-Path $PSScriptRoot 'json_to_typed_obb_dataset.py'
$cache = Join-Path $PSScriptRoot 'cst_airfoil_code_cache.pt'
$flyingWingSeed = 20260711
$conventionalCanardSeed = 20260727

foreach ($path in @(
    $condaExe,
    $mymlPython,
    $flyingWingGenerator,
    $conventionalCanardGenerator,
    $converter,
    $cache
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

function Get-DatasetPath([string]$layout, [string]$field) {
    $path = & $mymlPython -c "import project_paths; print(project_paths.AIRCRAFT_DATASET_SPECS['$layout']['$field'])"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read $field for $layout from project_paths.py"
    }
    return $path.Trim()
}

function Reset-DatasetDirectory([string]$datasetDirectory) {
    if ($approvedDatasetDirs -notcontains $datasetDirectory) {
        throw "Refusing to clear unapproved dataset directory: $datasetDirectory"
    }
    if (Test-Path -LiteralPath $datasetDirectory) {
        $resolvedDirectory = (Resolve-Path -LiteralPath $datasetDirectory).Path
        if ($approvedDatasetDirs -notcontains $resolvedDirectory) {
            throw "Resolved dataset directory is not approved: $resolvedDirectory"
        }
        Remove-Item -LiteralPath $resolvedDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $datasetDirectory | Out-Null
}

function Get-SampleCount([string]$moduleName) {
    $count = & $mymlPython -c "import sys; sys.path.insert(0, 'data'); import $moduleName as common; print(common.DATASET_SAMPLE_COUNT)"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read DATASET_SAMPLE_COUNT from $moduleName"
    }
    return [int]$count
}

Push-Location $repoRoot
try {
    $flyingWingOutputDir = Get-DatasetPath 'flying_wing' 'directory'
    $flyingWingDataset = Get-DatasetPath 'flying_wing' 'dataset'
    $conventionalCanardOutputDir = Get-DatasetPath 'conventional_canard' 'directory'
    $conventionalCanardDataset = Get-DatasetPath 'conventional_canard' 'dataset'
    $approvedDatasetDirs = @($flyingWingOutputDir, $conventionalCanardOutputDir)

    Reset-DatasetDirectory $flyingWingOutputDir
    Reset-DatasetDirectory $conventionalCanardOutputDir

    $condaHook = & $condaExe 'shell.powershell' 'hook' | Out-String
    Invoke-Expression $condaHook

    conda activate vsppytools
    $flyingWingCount = Get-SampleCount 'aircraft_dataset_common'
    & python $flyingWingGenerator '--output-dir' $flyingWingOutputDir '--count' $flyingWingCount '--seed' $flyingWingSeed
    if ($LASTEXITCODE -ne 0) {
        throw "Flying-wing OpenVSP/JSON generation failed with exit code $LASTEXITCODE"
    }

    $conventionalCanardCount = Get-SampleCount 'conventional_canard_dataset_common'
    & python $conventionalCanardGenerator '--output-dir' $conventionalCanardOutputDir '--count' $conventionalCanardCount '--seed' $conventionalCanardSeed
    if ($LASTEXITCODE -ne 0) {
        throw "Conventional/canard OpenVSP/JSON generation failed with exit code $LASTEXITCODE"
    }

    conda activate myml
    & python $converter '--input-dir' $flyingWingOutputDir '--output' $flyingWingDataset '--expected-count' $flyingWingCount
    if ($LASTEXITCODE -ne 0) {
        throw "Flying-wing JSON-to-PT conversion failed with exit code $LASTEXITCODE"
    }

    & python $converter '--input-dir' $conventionalCanardOutputDir '--output' $conventionalCanardDataset '--expected-count' $conventionalCanardCount
    if ($LASTEXITCODE -ne 0) {
        throw "Conventional/canard JSON-to-PT conversion failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
