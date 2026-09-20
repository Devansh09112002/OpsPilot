<#
.SYNOPSIS
    Collect the deployment credentials with hidden input and store them in
    OpsPilot's gitignored .env.

.DESCRIPTION
    Prompts for each credential with masked input, writes it into .env, and
    prints only a length and a 4-character tail so a truncated paste is
    visible without disclosing the secret.

    Run this in your own terminal: an agent's shell has stdin closed, so an
    interactive prompt cannot be driven from there.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File infra\setup_secrets.ps1
#>
[CmdletBinding()]
param(
    # Skip a credential you have already stored.
    [string[]]$Only
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $root '.env'

if (-not (Test-Path $envPath)) {
    $example = Join-Path $root '.env.example'
    if (Test-Path $example) { Copy-Item $example $envPath }
    else { New-Item -ItemType File -Path $envPath | Out-Null }
    Write-Host "Created .env" -ForegroundColor DarkGray
}

$wanted = @(
    @{ Key = 'SUPABASE_ACCESS_TOKEN'
       Hint = 'supabase.com/dashboard/account/tokens  (starts sbp_)' }
    @{ Key = 'RENDER_API_KEY'
       Hint = 'dashboard.render.com/u/settings#api-keys  (starts rnd_)' }
)
if ($Only) { $wanted = $wanted | Where-Object { $Only -contains $_.Key } }

function Set-EnvValue {
    param([string]$Path, [string]$Name, [string]$Value)
    $lines = @(Get-Content $Path)
    $found = $false
    $out = foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=") { $found = $true; "$Name=$Value" }
        else { $line }
    }
    if (-not $found) { $out = @($out) + @("$Name=$Value") }
    Set-Content -Path $Path -Value $out -Encoding utf8
}

Write-Host ""
Write-Host "OpsPilot deployment credentials" -ForegroundColor Cyan
Write-Host "Input is hidden. Values go only into .env, which is gitignored." -ForegroundColor DarkGray
Write-Host ""

$stored = @()
foreach ($item in $wanted) {
    Write-Host "$($item.Key)" -ForegroundColor White
    Write-Host "  from $($item.Hint)" -ForegroundColor DarkGray
    $secure = Read-Host -Prompt "  paste it" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }

    if ([string]::IsNullOrWhiteSpace($value)) {
        Write-Host "  skipped (nothing entered)" -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    $value = $value.Trim()
    Set-EnvValue -Path $envPath -Name $item.Key -Value $value
    $tail = if ($value.Length -ge 4) { $value.Substring($value.Length - 4) } else { '****' }
    Write-Host "  stored ($($value.Length) characters, ends '$tail')" -ForegroundColor Green
    Write-Host ""
    $stored += $item.Key
}

if ($stored.Count -eq 0) {
    Write-Host "Nothing was stored." -ForegroundColor Yellow
    exit 1
}

Write-Host "Stored: $($stored -join ', ')" -ForegroundColor Green
Write-Host ""
Write-Host "Done. Tell the agent the credentials are in place and it will" -ForegroundColor Cyan
Write-Host "run the deployment." -ForegroundColor Cyan
