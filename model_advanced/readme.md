# Minimalny własny model AI (PyTorch)

## Instalacja
```bash
pip install -r requirements.txt
```

## Przygotowanie danych
W katalogu `data/` powinien znaleźć się plik `train.csv`.
Ostatnia kolumna to etykieta (klasa), reszta to cechy.

## Trening modelu
```bash
cd src
python train.py
```
Wytrenowany model znajdziesz jako `model.pth` w katalogu głównym.
Możesz rozbudować sieć, dodać walidację, inne architektury, augmentację danych itd.
