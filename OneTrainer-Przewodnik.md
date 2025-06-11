
# OneTrainer – Przewodnik krok po kroku

## Wstęp
OneTrainer to nowoczesny, szybki i minimalistyczny framework do trenowania LoRA na Stable Diffusion XL, inspirowany Kohya_ss, ale uproszczony do maksimum.

---

## Wymagania wstępne
- Python 3.10+
- Pytorch 2.0+
- CUDA 11.8+
- Karta NVIDIA z min. 12 GB VRAM (zalecane 24 GB)

---

## Instalacja

```bash
git clone https://github.com/ExponentialML/OneTrainer.git
cd OneTrainer
pip install -r requirements.txt
```

---

## Przygotowanie danych

1. Umieść zdjęcia do treningu w katalogu `data/`.
2. Jeśli chcesz, przygotuj plik `captions.txt` (opcjonalnie, jeśli każde zdjęcie ma inne promptowanie).

---

## Trening

Przykład komendy:

```bash
python train_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="data" \
  --output_dir="lora_output" \
  --resolution=1024 \
  --train_batch_size=2 \
  --max_train_steps=2000 \
  --learning_rate=1e-4
```

**Najważniejsze parametry:**

- `--pretrained_model_name_or_path` – ścieżka do modelu SDXL lub checkpoint.
- `--train_data_dir` – katalog ze zdjęciami treningowymi.
- `--output_dir` – gdzie zapisać LoRA.
- `--resolution` – rozdzielczość zdjęć.
- `--max_train_steps` – liczba kroków treningowych.

---

## Generowanie obrazów z LoRA

Po treningu LoRA możesz dodać wagę do Fooocus, Automatic1111 lub ComfyUI (zgodnie z wybranym workflow).

**Przykładowy prompt do SDXL z LoRA:**

```
<lora:twoja_lora:0.6> piękna kobieta w stylu portretowym, Canon EOS 5D, soft lighting
```

---

## FAQ

> **Czy OneTrainer obsługuje trening tekstur lub stylów?**
> Tak, wystarczy przygotować odpowiedni zestaw zdjęć.
>
> **Jak sprawdzić wagę LoRA?**
> W narzędziach typu Fooocus, Automatic1111 – po prostu załaduj wagę jak inne LoRA.
>
> **Czy muszę robić captions?**
> Tylko jeśli chcesz mieć różne promptowanie do różnych zdjęć. Przy prostych LoRA można pominąć.

---

## Linki

- [Repozytorium OneTrainer (GitHub)](https://github.com/ExponentialML/OneTrainer)
- [Przykładowe LoRA](https://civitai.com/)

---

**Autor**: ExponentialML | Przewodnik: ChatGPT
