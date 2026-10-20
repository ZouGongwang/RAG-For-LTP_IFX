# Windows setup. Usage (PowerShell):  .\scripts\setup.ps1 [-Lite]
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
param([switch]$Lite)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Behind a company proxy add e.g.:  --proxy http://proxy.company.com:8080
if ($Lite) {
    .\.venv\Scripts\pip.exe install -e ".[cloud,dev]"
    Write-Host "Lite install: set 'parsing.backend: simple' in config.yaml"
} else {
    .\.venv\Scripts\pip.exe install -e ".[all]"
}
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Write-Host ""
Write-Host "Next steps:"
Write-Host "  Edit .env: set LLM_API_KEY (LiteLLM). Add SSL_CERT_FILE if corporate TLS requires it."
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m rag doctor"
Write-Host "  python -m rag ingest"
Write-Host "  python -m rag ask"
Write-Host "  (Optional local fallback: ollama pull deepseek-r1:8b; ollama pull llava:7b,"
Write-Host "   then set llm.provider / vision.provider to ollama in config.yaml)"
