<#
.SYNOPSIS
    Store a secret in OpsPilot's gitignored .env without ever displaying it.

.DESCRIPTION
    Prompts with masked input, writes the value into .env (replacing any
    existing entry for that key), and prints only a confirmation. The value
    is never echoed, never written to a tracked file, and never placed on the
    command line where a shell history would capture it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File infra\set_secret.ps1 SUPABASE_ACCESS_TOKEN
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        'SUPABASE_ACCESS_TOKEN',
        'RENDER_API_KEY',
        'GEMINI_API_KEY',
        'SUPABASE_DB_PASSWORD',
        'DATABASE_URL'
    )]
    [string]$Name
)

$ErrorActionPreference = 'Stop'

$envPath = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
if (-not (Test-Path $envPath)) {
    $example = Join-Path (Split-Path -Parent $PSScriptRoot) '.env.example'
    if (Test-Path $example) {
        Copy-Item $example $envPath
        Write-Host "Created .env from .env.example"
    } else {
        New-Item -ItemType File -Path $envPath | Out-Null
    }
}

$secure = Read-Host -Prompt "Paste $Name (input is hidden)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ([string]::IsNullOrWhiteSpace($value)) {
    Write-Host "Nothing entered; .env was not changed." -ForegroundColor Yellow
    exit 1
}
$value = $value.Trim()

# Replace an existing entry in place, or append a new one.
$lines = @(Get-Content $envPath)
$found = $false
$out = foreach ($line in $lines) {
    if ($line -match "^\s*$([regex]::Escape($Name))\s*=") {
        $found = $true
        "$Name=$value"
    } else {
        $line
    }
}
if (-not $found) { $out = @($out) + @("$Name=$value") }

Set-Content -Path $envPath -Value $out -Encoding utf8

# Confirm without revealing. A length and a 4-character tail is enough to spot
# a truncated paste without disclosing the secret.
$tail = if ($value.Length -ge 4) { $value.Substring($value.Length - 4) } else { '****' }
Write-Host ""
Write-Host "Stored $Name in .env  ($($value.Length) characters, ends '$tail')" -ForegroundColor Green
Write-Host ".env is gitignored and is never committed." -ForegroundColor DarkGray
