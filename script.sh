#!/bin/bash

set -e

echo "---------------"
echo "NBA Predictions"
echo "---------------"

log_info() {
    echo "[INFO] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warning "Venv nie jest aktywne"
    
    if [ -d "venv" ]; then
        log_info "Aktywuję środowisko wirtualne"
        source venv/bin/activate
    else
        log_error "Utwórz środowisko wirtualne: python3 -m venv venv"
        exit 1
    fi
fi

echo ""
log_info "Krok 1: Zbieranie danych NBA"
python3 main.py
if [ $? -ne 0 ]; then
    log_error "Błąd podczas zbierania danych NBA"
    exit 1
fi
log_info "Dane NBA zaktualizowane"

echo ""
log_info "Krok 2: Pobieranie kursów bukmacherskich"
if [ -f "betting_data/betting_main.py" ]; then
    if [ ! -f "betting_data/parameters.json" ]; then
        log_warning "Brak pliku betting_data/parameters.json"
        cat > betting_data/parameters.json << 'EOF'
{
    "request": {
        "sport": "basketball_nba",
        "region": "us",
        "market": "h2h",
        "format": "decimal"
    }
}
EOF
    fi
    
    cd betting_data
    python3 betting_main.py
    betting_exit_code=$?
    cd ..
    
    if [ $betting_exit_code -eq 0 ]; then
        log_info "Kursy bukmacherskie pobrane"
    else
        log_warning "Błąd podczas pobierania kursów"
    fi
else
    log_warning "Plik betting_data/betting_main.py nie znaleziony"
fi

echo ""
log_info "Krok 3: Trenowanie modeli ML"
python3 master_train_pipeline.py
if [ $? -ne 0 ]; then
    log_error "Błąd podczas trenowania"
    exit 1
fi
log_info "Modele ML wytrenowane"

echo ""
log_info "Krok 4: Porównanie z kursami bukmacherskimi"
if [ -f "betting_comparison.py" ]; then
    python3 betting_comparison.py
    if [ $? -eq 0 ]; then
        log_info "Porównanie z kursami zakończone"
    else
        log_warning "Błąd podczas porównania z kursami (mo"
    fi
fi

echo ""
echo "--------------------"
log_info "Zakończono pomyślnie"
echo "--------------------"
echo ""
log_info "Wygenerowano pliki:"
echo "  - Modele ML: models/*.joblib"
echo "  - Baza danych: data/nba_history.db"
echo "  - Features: data/ml_*.csv"
echo "  - Logi: logs/*.log"
echo ""
log_info "Aby uruchomić serwer Django:"
echo "  cd nba_prophecy_website"
echo "  python3 manage.py migrate"
echo "  python3 manage.py runserver"
echo ""