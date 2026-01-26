$ErrorActionPreference = "Stop"

Write-Host "---------------"
Write-Host "NBA Predictions"
Write-Host "---------------"


function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

if (-not $env:VIRTUAL_ENV) {
    Write-Warning-Custom "Venv nie jest aktywne"
    
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Info "Aktywuję środowisko wirtualne"
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Error-Custom "Utwórz środowisko wirtualne: python -m venv venv"
        exit 1
    }
}

try {
    Write-Host ""
    Write-Info "Krok 1: Zbieranie danych NBA"
    python main.py
    if ($LASTEXITCODE -ne 0) { throw "Błąd podczas zbierania danych NBA" }
    Write-Info "Dane NBA zaktualizowane"

    Write-Host ""
    Write-Info "Krok 2: Pobieranie kursów bukmacherskich"
    if (Test-Path "betting_data\betting_main.py") {
        try {
            if (-not (Test-Path "betting_data\parameters.json")) {
                Write-Warning-Custom "Brak pliku betting_data\parameters.json"
                $defaultParams = @{
                    request = @{
                        sport = "basketball_nba"
                        region = "us"
                        market = "h2h"
                        format = "decimal"
                    }
                } | ConvertTo-Json -Depth 10
                $defaultParams | Out-File -FilePath "betting_data\parameters.json" -Encoding UTF8
            }
            
            Push-Location betting_data
            python betting_main.py
            $bettingExitCode = $LASTEXITCODE
            Pop-Location
            
            if ($bettingExitCode -eq 0) {
                Write-Info "Kursy bukmacherskie pobrane"
            } else {
                Write-Warning-Custom "Błąd podczas pobierania kursów"
            }
        } catch {
            Write-Warning-Custom "Błąd podczas pobierania kursów: $_"
        }
    } else {
        Write-Warning-Custom "Plik betting_data\betting_main.py nie znaleziony"
    }

    Write-Host ""
    Write-Info "Krok 3: Trenowanie modeli ML"
    python master_train_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw "Błąd podczas trenowania" }
    Write-Info "Modele ML wytrenowane"

    Write-Host ""
    Write-Info "Krok 4: Porównanie z kursami bukmacherskimi"
    if (Test-Path "betting_comparison.py") {
        try {
            python betting_comparison.py
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Porównanie z kursami zakończone"
            } else {
                Write-Warning-Custom "Błąd podczas porównania z kursami"
            }
        } catch {
            Write-Warning-Custom "Błąd podczas porównania z kursami"
        }
    }

    Write-Host ""
    Write-Info "Krok 5: Aktualizacja wyników meczów i zakładów"
    Push-Location nba_prophecy_website
    try {
        python manage.py update_bets
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Zakłady zaktualizowane"
        } else {
            Write-Warning-Custom "Błąd podczas aktualizacji zakładów"
        }
    } catch {
        Write-Warning-Custom "Błąd podczas aktualizacji zakładów: $_"
    }        
    Pop-Location

    Write-Host ""
    Write-Host "--------------------"
    Write-Info "Zakończono pomyślnie"
    Write-Host "--------------------"
    Write-Host ""
    Write-Info "Wygenerowano pliki:"
    Write-Host "  - Modele ML: models\*.joblib"
    Write-Host "  - Baza danych: data\nba_history.db"
    Write-Host "  - Features: data\ml_*.csv"
    Write-Host "  - Logi: logs\*.log"
    Write-Host ""
    Write-Info "Aby uruchomić serwer Django:"
    Write-Host "  cd nba_prophecy_website"
    Write-Host "  python manage.py migrate"
    Write-Host "  python manage.py runserver"
    Write-Host ""

} catch {
    Write-Host ""
    Write-Error-Custom "Pipeline zakończony z błędem: $_"
    exit 1
}