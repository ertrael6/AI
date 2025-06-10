# Fooocus – Szczegółowa, techniczna dokumentacja instalacji i obsługi (Windows)

Repozytorium: Kompletny przewodnik do instalacji, konfiguracji, obsługi oraz debugowania Fooocus (Stable Diffusion XL) w środowisku offline, z naciskiem na pełne zrozumienie działania każdego kroku.

---

## Spis treści

1. [Wymagania sprzętowe i systemowe](#1-wymagania-sprzętowe-i-systemowe)
2. [Python – instalacja i konfiguracja środowiska](#2-python--instalacja-i-konfiguracja-środowiska)
3. [Fooocus – pobranie, struktura katalogów](#3-fooocus--pobranie-struktura-katalogów)
4. [Manualna instalacja zależności (biblioteki Python, CUDA, Torch)](#4-manualna-instalacja-zależności-biblioteki-python-cuda-torch)
5. [Pierwsze uruchomienie – co się dzieje „pod maską”](#5-pierwsze-uruchomienie--co-się-dzieje-pod-maską)
6. [Konfiguracja modeli SDXL, LoRA, VAE, Upscaler – struktura plików](#6-konfiguracja-modeli-sdxl-lora-vae-upscaler--struktura-plików)
7. [Debugowanie i monitoring – gdzie szukać logów, rozwiązywanie problemów](#7-debugowanie-i-monitoring--gdzie-szukać-logów-rozwiązywanie-problemów)
8. [Przykłady promptów i workflowów](#8-przykłady-promptów-i-workflowów)
9. [Przydatne linki, dokumentacja i narzędzia](#9-przydatne-linki-dokumentacja-i-narzędzia)

---

## 1. Wymagania sprzętowe i systemowe

- **Karta graficzna NVIDIA** (min. 6 GB VRAM, optymalnie RTX 20XX lub nowsza, wsparcie CUDA 11.8+)
- **RAM:** min. 16 GB
- **CPU:** dowolny nowszy, 4 rdzenie+
- **System:** Windows 10/11, 64-bit
- **Dysk:** min. 20–30 GB wolnego miejsca (modele mogą zajmować dużo przestrzeni)
- **Sterowniki NVIDIA:** aktualne, wsparcie CUDA (sprawdź wersję: Panel Sterowania NVIDIA)

---

## 2. Python – instalacja i konfiguracja środowiska

### **A. Instalacja Pythona**

1. Wejdź na: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Pobierz **Python 3.10.x** lub **3.11.x**.  
   Nie instaluj 3.12 (często nie działa z Fooocus/SDXL)!
3. Uruchom instalator.  
   **WAŻNE:** ZAZNACZ “Add Python to PATH” (inaczej `python` nie będzie działał w terminalu).
4. Po instalacji sprawdź wersję:  
   - Otwórz **cmd**, wpisz:
     ```
     python --version
     ```
   - Oczekiwany wynik: `Python 3.10.X` lub `3.11.X`

### **B. Instalacja PIP i aktualizacja**

- PIP (menadżer pakietów Pythona) powinien być zainstalowany automatycznie.
- W razie problemów:  
  ```
  python -m ensurepip --upgrade
  python -m pip install --upgrade pip
  ```

---

## 3. Fooocus – pobranie, struktura katalogów

### **A. Pobranie i rozpakowanie**

1. Idź na: [https://github.com/lllyasviel/Fooocus/releases](https://github.com/lllyasviel/Fooocus/releases)
2. Pobierz **Fooocus-win.zip** (lub odpowiedni dla twojego systemu).
3. Rozpakuj do katalogu, np. `C:\AI\Fooocus`.

### **B. Struktura katalogów po rozpakowaniu**
```
Fooocus/
│
├── models/
│   ├── checkpoints/        ← tu trafiają pliki modeli SDXL (.safetensors, .ckpt)
│   ├── loras/              ← tu wrzucasz LoRA (.safetensors)
│   ├── vae/                ← tu pliki VAE (.vae.pt, .safetensors)
│   ├── upscale_models/     ← tu pliki upscalerów (.pth)
│
├── outputs/                ← tu pojawią się wygenerowane obrazy
├── extensions/             ← dodatkowe pluginy
├── requirements_versions.txt  ← lista wersji bibliotek Python wymaganych przez Fooocus
├── run.bat                 ← uruchomienie Fooocus z domyślną konfiguracją (NVIDIA)
├── run_cpu.bat             ← uruchomienie na CPU (wolne, do testów lub gdy brak NVIDIA)
├── entry_with_update.py    ← główny plik startowy Fooocusa
├── settings.yaml           ← konfiguracja ustawień (można edytować ręcznie)
└── ... (inne pliki i foldery)
```

---

## 4. Manualna instalacja zależności (biblioteki Python, CUDA, Torch)

### **A. Instalacja zależności globalnych**

W katalogu Fooocus otwórz terminal (`cmd`):

#### **1. Aktualizacja pip**
```sh
python -m pip install --upgrade pip
```
#### **2. Instalacja PyTorch z CUDA**

- Jeśli masz kartę NVIDIA i chcesz maksymalnej wydajności:
    ```sh
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
- Jeśli używasz CPU (wolno!):
    ```sh
    python -m pip install torch torchvision torchaudio
    ```
- **Co tu się dzieje?**
    - PyTorch (torch) to podstawa większości modeli AI.
    - `--index-url ...` wymusza pobranie wersji z obsługą CUDA 11.8, niezbędnej dla nowych kart NVIDIA.

#### **3. Instalacja wszystkich pozostałych zależności Fooocusa**
```sh
python -m pip install -r requirements_versions.txt
```
- Ten plik wymusza dokładnie takie wersje bibliotek jakich oczekuje Fooocus (np. numpy, transformers, pillow, requests itd.).
- Dzięki temu nie ma konfliktów wersji.

---

## 5. Pierwsze uruchomienie – co się dzieje „pod maską”

### **A. Uruchamianie Fooocus**

- Kliknij `run.bat` (lub `run_cpu.bat` dla CPU)  
  **LUB** uruchom w terminalu:
  ```
  python entry_with_update.py
  ```

### **B. Co robi Fooocus podczas startu?**
1. **Sprawdza dostępność wymaganych bibliotek i wersji** (może je doinstalować jeśli brak).
2. **Łączy się z internetem (przy pierwszym uruchomieniu)** i pobiera:
    - Domyślny model SDXL (zazwyczaj Juggernaut XL lub inny).
    - Domyślne pliki LoRA, VAE, upscalery.
3. **Tworzy folder `outputs/`** na wygenerowane obrazy.
4. **Uruchamia serwer lokalny** (domyślnie na http://localhost:7865).
5. **Sprawdza, czy istnieją dodatkowe modele w katalogach `models/`** – jeśli tak, dołącza je do listy wyboru w GUI.
6. **Weryfikuje konfigurację CUDA/Torch** – wyświetla logi czy wykryto GPU i czy obsługuje CUDA.
7. **Logi pojawiają się w konsoli (cmd)** – przy błędach warto je skopiować i szukać rozwiązania.

---

## 6. Konfiguracja modeli SDXL, LoRA, VAE, Upscaler – struktura plików

### **A. Modele SDXL**

- **Gdzie znaleźć:**  
  [Civitai – SDXL](https://civitai.com/models?tag=sdxl)  
  [Hugging Face – SDXL](https://huggingface.co/models?search=sdxl)
- **Gdzie wrzucać:**  
  `Fooocus/models/checkpoints/`
- **Format:** `.safetensors` (zalecane) lub `.ckpt`
- **W GUI**: pojawią się w sekcji wyboru modelu po restarcie Fooocus

### **B. LoRA**

- **Gdzie znaleźć:**  
  [Civitai – LoRA SDXL](https://civitai.com/models?tag=lora&tag=sdxl)
- **Gdzie wrzucać:**  
  `Fooocus/models/loras/`
- **Format:** `.safetensors`
- **W GUI**: sekcja Model → wybór LoRA; można mieszać kilka naraz, suwakami ustawiać wagę

### **C. VAE**

- **Gdzie znaleźć:**  
  [stabilityai/sdxl-vae na Hugging Face](https://huggingface.co/stabilityai/sdxl-vae)
- **Gdzie wrzucać:**  
  `Fooocus/models/vae/`
- **Format:** `.vae.pt` lub `.safetensors`
- **W GUI:** nie zawsze jest widoczny wybór – domyślnie ładowany na starcie, można podmienić plik

### **D. Upscale Models**

- **Gdzie znaleźć:**  
  [RealESRGAN models](https://github.com/xinntao/Real-ESRGAN)
- **Gdzie wrzucać:**  
  `Fooocus/models/upscale_models/`
- **Format:** `.pth`
- **W GUI:** opcja zwiększania rozdzielczości po wygenerowaniu obrazu

---

## 7. Debugowanie i monitoring – gdzie szukać logów, rozwiązywanie problemów

### **A. Logi w konsoli**

- Wszystkie komunikaty Fooocusa wyświetlane są w otwartym oknie konsoli (cmd).
- **Główne etapy:**
    - Sprawdzanie bibliotek
    - Informacja o wykrytej karcie graficznej i CUDA
    - Pobieranie modeli
    - Błędy związane z Torch, CUDA, modelem, brakiem pamięci, itp.

### **B. Typowe błędy i rozwiązania**

| Problem                           | Przyczyna & Rozwiązanie                                    |
|------------------------------------|------------------------------------------------------------|
| Fooocus nie startuje, brak GUI     | Sprawdź wersję Pythona, pip, zależności, logi w konsoli    |
| Błąd CUDA / Torch                  | Nieprawidłowa wersja Torch/Python, konflikt CUDA           |
| Modele nie widoczne                | Zły katalog, zły format pliku (nie SDXL!), brak restartu   |
| “Out of Memory”                    | Za mało VRAM/RAM, zmniejsz rozdzielczość, mniej obrazów    |
| Brak modelu VAE                    | Pobierz odpowiedni VAE, wrzuć do `models/vae/`             |
| Brak LoRA na liście                | Nie ten katalog, zły format pliku, potrzebny restart       |
| Fooocus zawiesza się przy starcie  | Usuń plik `settings.yaml` (czasem uszkodzony), spróbuj ponownie |

---

## 8. Przykłady promptów i workflowów

### **Prompt typowy:**
```
A photorealistic portrait of a young woman in the forest, natural light, soft focus, Canon EOS 5D, SDXL, masterpiece
```

### **Workflow – zmiana modelu, LoRA i VAE:**

1. Pobierz model SDXL z Civitai, wrzuć do `models/checkpoints/`
2. Pobierz LoRA z Civitai, wrzuć do `models/loras/`
3. Pobierz VAE, wrzuć do `models/vae/`
4. Uruchom Fooocus (`run.bat`)
5. Wybierz model, LoRA, prompt, kliknij Generate

---

## 9. Przydatne linki, dokumentacja i narzędzia

- [Fooocus GitHub](https://github.com/lllyasviel/Fooocus)
- [Fooocus FAQ](https://github.com/lllyasviel/Fooocus#faq)
- [Civitai SDXL Models](https://civitai.com/models?tag=sdxl)
- [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)
- [PyTorch download](https://pytorch.org/get-started/locally/)
- [RealESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [Fooocus Discord](https://discord.gg/fooocus)
- [Lexica – przykładowe prompty](https://lexica.art/)

---

> Ten dokument możesz śmiało kopiować do repozytorium! Jeśli chcesz – mogę przygotować wersję z obrazkami, pokazami katalogów, czy jeszcze bardziej „dev-friendly” (np. osobne sekcje dla Windows/Linux/Mac).
