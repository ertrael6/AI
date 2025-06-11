# OneTrainer – Kompletny Przewodnik (Polski)

## Wprowadzenie

**OneTrainer (OT)** to kompleksowe narzędzie do trenowania modeli generatywnych (Stable Diffusion) z przyjaznym interfejsem graficznym (GUI) oraz trybem konsolowym (CLI). Pozwala na pełne dostrajanie modeli (fine-tuning), trenowanie LoRA (Low-Rank Adaptation) oraz wgrywanie tekstowych osadzeń (textual inversion). Dzięki OneTrainer możesz trenować modele na bazie różnych wersji Stable Diffusion (v1.5, v2.x, SDXL, a nawet architekturę Flux) w prosty sposób – skryptem instalacyjnym lub z użyciem wirtualnego środowiska Pythona.

Przewodnik oparty jest na oficjalnej dokumentacji (wiki GitHub OneTrainer) – masz pewność aktualności na czerwiec 2025.  
Sekcje oznaczone jako **Screenshot** zawierają miejsce na zrzuty ekranu (konfiguracji, wyników oraz uruchomienia programu), aby zilustrować proces.

---

## Spis treści

- [Wprowadzenie](#wprowadzenie)
- [Instalacja OneTrainer (GUI i CLI)](#instalacja-onetrainer-gui-i-cli)
- [Uruchomienie programu – tryb graficzny i konsolowy](#uruchomienie-programu--tryb-graficzny-i-konsolowy)
- [Interfejs graficzny OneTrainer – przegląd zakładek](#interfejs-graficzny-onetrainer--przegląd-zakładek)
- [Zakładka General (Ogólne)](#zakładka-general-ogólne)
- [Zakładka Model](#zakładka-model)
- [Zakładka Data (Dane)](#zakładka-data-dane)
- [Zakładka Concepts (Zbiór danych)](#zakładka-concepts-zbiór-danych)
- [Zakładka Training (Trening)](#zakładka-training-trening)
- [Zakładka LoRA](#zakładka-lora)
- [Zakładka Sampling i Backup](#zakładka-sampling-i-backup)
- [Zakładka Tools (Narzędzia)](#zakładka-tools-narzędzia)
- [Przygotowanie danych do treningu](#przygotowanie-danych-do-treningu)
- [Obrazy i podpisy (.txt)](#obrazy-i-podpisy-txt)
- [Augmentacje obrazów](#augmentacje-obrazów)
- [Proporcje obrazu i bucketing (Aspect Ratio Buckets)](#proporcje-obrazu-i-bucketing-aspect-ratio-buckets)
- [Trenowanie modelu LoRA – przykłady](#trenowanie-modelu-lora--przykłady)
  - [LoRA na bazie SDXL](#lora-na-bazie-sdxl)
  - [LoRA na bazie modelu Flux](#lora-na-bazie-modelu-flux)
- [Zastosowanie wytrenowanych modeli w innych narzędziach](#zastosowanie-wytrenowanych-modeli-w-innych-narzędziach)
- [Tryb CLI – skrypty i ich zastosowania](#tryb-cli--skrypty-i-ich-zastosowania)
- [Zaawansowane opcje OneTrainer](#zaawansowane-opcje-onetrainer)
- [Praktyczne porady i najlepsze praktyki](#praktyczne-porady-i-najlepsze-praktyki)

---

## Instalacja OneTrainer (GUI i CLI)

### Wymagania wstępne

- **Python:** Wersja 3.10–3.12 (nowszych niż 3.13 nie wspiera)
- **PyTorch:** 2.6.0+ kompatybilny z Twoim akceleratorem (CUDA 11.8+ dla Nvidia, ROCm 6.2.4+ dla AMD)
- **Wolne miejsce na dysku:** min. 7 GB
- **RAM (dla offloadingu na CPU):** zalecane 64 GB (minimum 32 GB, mogą wystąpić błędy OOM)

### Krok 1: Pobranie kodu źródłowego

Najlepiej sklonować repozytorium GitHub OneTrainer:

```bash
git clone https://github.com/Nerogar/OneTrainer.git


To pobierze wszystkie pliki programu na Twój dysk.

### Krok 2: Instalacja zależności

OneTrainer oferuje skrypt automatycznej instalacji. Po sklonowaniu repozytorium:

- **Windows:** Uruchom (poprzez dwuklik) plik `install.bat` w folderze OneTrainer.
- **Linux/macOS:** Nadaj prawa wykonywalności `install.sh` i uruchom go:

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

Skrypt utworzy środowisko wirtualne Pythona, zainstaluje wymagane pakiety (PyTorch, Diffusers, itd.) oraz skonfiguruje OneTrainer. Całość trwa kilka minut.

#### Alternatywa – instalacja manualna

Możesz wykonać to ręcznie:

1. **Utwórz wirtualne środowisko:**
    ```bash
    cd OneTrainer
    python -m venv venv
    ```
2. **Aktywuj środowisko:**
    - Windows:
        ```bash
        venv\Scripts\activate
        ```
    - Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
3. **Zainstaluj wymagane pakiety:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Uruchomienie OneTrainer

OneTrainer oferuje zarówno interfejs graficzny (GUI), jak i narzędzia linii komend (CLI).

### Uruchomienie GUI

Po instalacji przejdź do folderu OneTrainer i uruchom GUI:

```bash
python gui.py
```

Interfejs graficzny pozwala na konfigurację i uruchamianie treningów w sposób przyjazny dla początkujących.

### Uruchomienie CLI

Możesz też korzystać z narzędzi CLI, uruchamiając:

```bash
python cli.py --help
```

Wyświetli to dostępne komendy i opcje.

---
```
