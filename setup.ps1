# SPDX-License-Identifier: GPL-3.0-only
# Installer only: this script never starts ComfyUI.
[CmdletBinding()]
param(
    [string]$ComfyRoot,
    [string]$Python,
    [string]$CacheSource,
    [string]$Gpu = '0',
    [switch]$BuildCache,
    [switch]$CheckOnly,
    [switch]$VerifyModels
)
$ErrorActionPreference = 'Stop'
try {
    if (-not $ComfyRoot) {
        $candidate = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
        if (Test-Path -LiteralPath (Join-Path $candidate 'comfy\sd.py')) {
            $ComfyRoot = $candidate
        } else {
            $ComfyRoot = (Read-Host 'ComfyUI directory (contains main.py)').Trim('"')
        }
    }
    $ComfyRoot = (Resolve-Path -LiteralPath $ComfyRoot).Path
    if (-not (Test-Path -LiteralPath (Join-Path $ComfyRoot 'comfy\sd.py'))) {
        throw 'Select the ComfyUI directory containing main.py and comfy\sd.py.'
    }
    if (-not $Python) {
        $parent = Split-Path $ComfyRoot -Parent
        $candidates = @(
            (Join-Path $parent 'python_embeded\python.exe'),
            (Join-Path $parent 'python_embedded\python.exe'),
            (Join-Path $ComfyRoot '.venv\Scripts\python.exe'),
            (Join-Path $parent '.venv\Scripts\python.exe'),
            (Join-Path $ComfyRoot 'venv\Scripts\python.exe')
        ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique
        if (@($candidates).Count -ne 1) {
            throw 'Cannot choose one isolated Python. Pass -Python <ComfyUI venv or embedded python.exe>. Never use system Python.'
        }
        $Python = @($candidates)[0]
    }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    $arguments = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'setup_env.py'), '--comfy-root', $ComfyRoot, '--gpu', $Gpu)
    if ($CacheSource) { $arguments += @('--cache-source', (Resolve-Path -LiteralPath $CacheSource).Path) }
    if ($BuildCache) { $arguments += '--build-cache' }
    if ($CheckOnly) { $arguments += '--check-only' }
    if ($VerifyModels) { $arguments += '--verify-models' }
    & $Python @arguments
    exit $LASTEXITCODE
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
