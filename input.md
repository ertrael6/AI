**OneTrainer** **–** **Kompletny** **Przewodnik** **(Polski)**
**Wprowadzenie**

OneTrainer (OT) to kompleksowe narzędzie do trenowania modeli
generatywnych (Stable Diffusion) z przyjaznym interfejsem graficznym
(GUI) oraz trybem konsolowym (CLI). Pozwala na pełne dostrajanie modeli
(fine-tuning), trenowanie LoRA (Low-Rank Adaptation) oraz wgrywanie
tekstowych osadzeń (textual inversion). 
Dzięki OneTrainer możesz trenować modele na bazie różnych wersji
Stable Diffusion (v1.5, v2.x, SDXL, a nawet architekturę Flux) w prosty
sposób – skryptem instalacyjnym lub z użyciem wirtualnego środowiska
Pythona. Poniższy przewodnik zawiera szczegółowe instrukcje instalacji (krok po
kroku), omówienie wszystkich zakładek i opcji GUI, przygotowanie danych
treningowych, prowadzenie treningu LoRA (na modelach SDXL i Flux) oraz
użycie zaawansowanych funkcji OneTrainer. Całość jest oparta o
**oficjalną** **dokumentację** (wiki GitHub OneTrainer) – dzięki temu
masz pewność aktualności informacji na czerwiec 2025.


**Spis** **treści**
## Spis treści

1. [Instalacja OneTrainer (GUI i CLI)](#instalacja-onetrainer-gui-i-cli)
2. [Uruchomienie programu – tryb graficzny i konsolowy](#uruchomienie-programu--tryb-graficzny-i-konsolowy)
3. [Interfejs graficzny OneTrainer – przegląd zakładek](#interfejs-graficzny-onetrainer--przeglad-zakladek)
4. [Zakładka General (Ogólne)](#zakladka-general-ogolne)
5. [Zakładka Model](#zakladka-model)
6. [Zakładka Data (Dane)](#zakladka-data-dane)
7. [Zakładka Concepts (Zbiór danych)](#zakladka-concepts-zbior-danych)
8. [Zakładka Training (Trening)](#zakladka-training-trening)
9. [Zakładka LoRA](#zakladka-lora)
10. [Zakładka Sampling i Backup](#zakladka-sampling-i-backup)
11. [Zakładka Tools (Narzędzia)](#zakladka-tools-narzedzia)
12. [Przygotowanie danych do treningu](#przygotowanie-danych-do-treningu)
13. [Obrazy i podpisy(.txt)](#obrazy-i-podpisytxt)
14. [Augmentacje obrazów](#augmentacje-obrazow)
15. [Proporcje obrazu i bucketing (Aspect Ratio Buckets)](#proporcje-obrazu-i-bucketing-aspect-ratio-buckets)
16. [Trenowanie modelu LoRA – przykłady](#trenowanie-modelu-lora--przyklady)
17. [LoRA na bazie SDXL](#lora-na-bazie-sdxl)
18. [LoRA na bazie modelu Flux](#lora-na-bazie-modelu-flux)
19. [Zastosowanie wytrenowanych modeli w innych narzędziach](#zastosowanie-wytrenowanych-modeli-w-innych-narzedziach)
20. [Tryb CLI – skrypty i ich zastosowania](#tryb-cli--skrypty-i-ich-zastosowania)
21. [Zaawansowane opcje OneTrainer](#zaawansowane-opcje-onetrainer)
22. [Ofloading do RAM (przenoszenie obciążenia na pamięć RAM)](#ofloading-do-ram-przenoszenie-obciazenia-na-pamiec-ram)
23. [Precyzja obliczeń (precision) i typy danych](#precyzja-obliczen-precision-i-typy-danych)
24. [Różne typy LoRA (LoRA, LoHa, DoRA)](#rozne-typy-lora-lora-loha-dora)
25. [Wybór trenowanych warstw modelu](#wybor-trenowanych-warstw-modelu)
26. [Praktyczne porady i najlepsze praktyki](#praktyczne-porady-i-najlepsze-praktyki)



## Instalacja OneTrainer (GUI i CLI)

**Wymagania** **wstępne:** OneTrainer wymaga zainstalowanego **Pythona**
**w** **wersji** **3.10–3.12** (nowszych niż 3.13 nie wspiera) oraz
**PyTorch** **2.6.0+** kompatybilnego z posiadaną akceleratorem (CUDA
11.8+ dla kart Nvidia lub ROCm 6.2.4+ dla kart AMD). Upewnij się, że masz co najmniej **7** **GB** **wolnego** **miejsca**
**dyskowego**, ponieważ instalacja i pliki modeli tyle zajmą. 
Jeśli planujesz korzystać z treningu z ofloadingiem na CPU, zalecane
jest 64 GB RAM (minimum 32 GB, ale mogą wystąpić błędy Out-Of-Memory).

**Krok** **1:** **Pobranie** **kodu** **źródłowego.** Najlepiej
sklonować repozytorium GitHub OneTrainer. Wykonaj w terminalu polecenie:

 git clone 

To pobierze wszystkie pliki programu na Twój dysk.

**Krok** **2:** **Instalacja** **zależności.** OneTrainer oferuje
*skrypt* *automatycznej* *instalacji*. Po sklonowaniu repozytorium:

 • **Windows:** Uruchom (poprzez dwuklik) plik install.bat w folderze OneTrainer.

 • **Linux/macOS:** Nadaj prawa wykonywalności install.sh i uruchom go
 '( ./install.sh ).'

Skrypt ten automatycznie utworzy środowisko wirtualne Pythona,
zainstaluje wymagane pakiety (w tym PyTorch, biblioteki HuggingFace
Diffusers itp.) oraz skonfiguruje OneTrainer. Dzięki temu nie musisz
ręcznie instalować zależności – wszystko dzieje się automatycznie w
kilku minutach.

**Alternatywa** **–** **instalacja** **manualna:** Jeżeli wolisz ręcznie
kontrolować środowisko, możesz wykonać następujące kroki (odpowiadające
temu, co robi skrypt):

 1. Utwórz wirtualne środowisko: przejdź do katalogu OneTrainer i
 wykonaj python '-m venv venv.'

 2. Aktywuj utworzone środowisko:

 3. Windows: 'venvScriptsactivate'

 4. Linux/macOS: 'source venv/bin/activate.'

 5. Zainstaluj wymagane pakiety: 'pip install -r requirements.txt'

 6. (Linux) Zainstaluj dodatkowe biblioteki systemowe, jeśli
 potrzebne: np. na Ubuntu brakująca może być biblioteka **libGL**,
 doinstalujesz ją komendą sudo apt-get install libgl1. (W Alpine Linux zainstaluj py3-tk , w Arch: tk – to zapewnia
 działanie interfejsu tkinter dla GUI).

Po poprawnej instalacji, w folderze OneTrainer pojawi się wirtualne
środowisko ( venv ) oraz wszystkie zależności. Możesz teraz przejść do
uruchomienia programu.

## Uruchomienie programu – tryb graficzny i konsolowy

OneTrainer oferuje dwa tryby działania:

 • **GUI** **(graficzny** **interfejs** **użytkownika):** Przyjazny
 interfejs okienkowy, pozwalający konfigurować trening za pomocą
 formularzy i przycisków.

 • **CLI** **(interfejs** **konsolowy):** Zestaw skryptów Pythona
 umożliwiających uruchamianie treningu i narzędzi z linii poleceń, co
 przydaje się przy automatyzacji, zdalnej pracy (np. serwery bez
 środowiska graficznego) lub bardziej zaawansowanej kontroli.

**Uruchomienie** **GUI:**

 • **Windows:** Uruchom plik start-ui.bat (np. dwukrotnie klikając go
 lub przez konsolę) w katalogu głównym OneTrainer.
 Spowoduje to otwarcie okienkowego interfejsu OT.

 • **Linux/macOS:** Wykonaj skrypt start-ui.sh ( ./start-ui.sh ) w
 katalogu OneTrainer – aplikacja graficzna powinna się uruchomić . Uwaga: na
 systemach Linux wymagane może być zainstalowanie pakietu tk (jeśli GUI
 nie startuje).

Po chwili powinna pojawić się główna **aplikacja** **OneTrainer** – okno
z menu zakładek (omówimy je w kolejnym rozdziale). *Jeśli* *GUI* *nie*
*uruchamia* *się*, sprawdź czy Python i zależności zostały poprawnie
zainstalowane (uruchom ponownie install.sh / install.bat ), a na
Linux/macOS czy masz zainstalowane tkinter/GUI (czasem trzeba
doinstalować). Również upewnij się, że posiadasz kartę graficzną
spełniającą wymagania (min. 8 GB VRAM zalecane dla SDXL/Flux).

**Uruchomienie** **CLI:**

Aby korzystać z trybu konsolowego, otwórz terminal/wiersz poleceń
**wewnątrz** **aktywowanego** **środowiska** **wirtualnego**
**OneTrainer** (tj. po activate – dzięki temu używany jest poprawny
Python i zainstalowane pakiety). W folderze OneTrainer znajduje się
katalog scripts , zawierający różne skrypty odpowiadające funkcjom
programu. Każdy skrypt możesz uruchomić za pomocą Pythona, np.:

 cd OneTrainer

 # aktywuj venv, jeśli jeszcze nie jest aktywny python
 scripts/train.py --help

Powyższe polecenie wyświetli dostępne opcje głównego skryptu
treningowego train.py . Podobnie, python scripts/generate_captions.py -h
pokaże opcje automatycznego generowania podpisów dla obrazów datasetu itd.

**Uwaga:** Wszystkie komendy muszą być wykonywane w aktywnym środowisku
wirtualnym, aby używać właściwych zależności.Jeśli używasz Linux/Unix, możesz zapoznać się z dokumentacją skryptów
startowych (launch scripts) dla wskazówek odnośnie uruchamiania
OneTrainer i skryptów w różnych systemach
*(Screenshot:* *Ekran* *startowy* *OneTrainer* *po* *uruchomieniu* *GUI*
*–* *miejsce* *na* *zrzut* *interfejsu* *aplikacji.)*

## Interfejs graficzny OneTrainer – przegląd zakładek

Po uruchomieniu GUI zobaczysz główne okno OneTrainer, które składa się z
menu **zakładek** (tabs) u góry oraz paneli z opcjami. Interfejs jest
dość prosty i uporządkowany: ustawienia treningu podzielono na sekcje
tematyczne, dostępne jako osobne zakładki. Warto zaznaczyć, że OneTrainer udostępnia również gotowe **preset**
**configurations** – w lewym górnym rogu znajdziesz menu wyboru
konfiguracji (domyślnie puste, możesz tam wybrać np. gotowe presety dla
LoRA SD1.5, LoRA SDXL itp., co automatycznie ustawi wiele opcji). Jako początkujący możesz skorzystać z tych presetów, aby mieć punkt
wyjścia.

Poniżej opisuje **każdą** **zakładkę** i znajdujące się w niej opcje:

## Zakładka General (Ogólne)

Zakładka **General** służy do ustawienia podstawowych ścieżek i
globalnych opcji treningu. Tutaj definiujemy katalogi robocze i
zachowanie programu podczas treningu. Główne opcje w tej zakładce to:

 • **Workspace** **Directory** – katalog roboczy (domyślnie
 workspace/run ). OneTrainer zapisuje tu wszelkie dane z przebiegu
 treningu: wygenerowane próbki (obrazy podglądowe), logi TensorBoard,
 kopie zapasowe modelu itp. Możesz utrzymywać jeden wspólny folder lub utworzyć osobny folder
 dla każdego projektu (druga opcja jest zalecana przy wielu
 równoległych projektach, by dane się nie mieszały).

 • **Cache** **Directory** – katalog cache (domyślnie
 workspace-cache/run ). Tutaj trafiają *przetworzone* *obrazy* *i*
 *teksty* w trakcie treningu (patrz: latent caching w zakładce Data).
 Zostaw domyślną ścieżkę, chyba że musisz umieścić cache na innym dysku
 (np. szybszym).

 • **Continue** **from** **last** **backup** – (domyślnie wyłączone)
 włączenie tej opcji spowoduje, że jeśli istnieje kopia zapasowa modelu
 w katalogu roboczym (np. po wcześniejszym przerwanym treningu),
 trening zostanie wznowiony od tej kopii. Używaj, gdy chcesz kontynuować trening zamiast zaczynać od nowa.

 • **Debug** **Mode** – (domyślnie off) tryb debugowania, w którym
 OneTrainer generuje dodatkowe dane diagnostyczne podczas treningu. W trybie debug otrzymasz np. obrazy przedstawiające porównanie
 przewidywania modelu vs rzeczywistego obrazu na krokach treningowych,
 co pomaga zrozumieć postępy. *Uwaga:* Obrazy debug generowane są
 poprzez VAE modelu z zachowanych tensorów, więc ich jakość może być
 niższa niż oryginałów, ale nadal są przydatne do analizy . Po włączeniu debug mode, musisz też podać **Debug** **Directory**
 (domyślnie debug ) – folder, gdzie te dane się zapiszą.

 • **TensorBoard** – (domyślnie włączone) czy uruchamiać serwer
 TensorBoard podczas treningu. TensorBoard umożliwia śledzenie wykresów strat (loss), ewentualnie
 podgląd próbek w trakcie treningu itp. OneTrainer integruje się z
 TensorBoard automatycznie – jeżeli opcja jest ON, po starcie treningu
 możesz kliknąć przycisk *“Tensorboard”* i obserwować statystyki w
 przeglądarce.

 • **Expose** **Tensorboard** – (domyślnie off) jeżeli włączysz,
 TensorBoard zostanie wystawiony na interfejs sieciowy (nie tylko
 *localhost*). Przydatne, gdy trenujesz zdalnie (np. serwer) i chcesz
 podejrzeć wykresy ze swojego komputera.

 • **Validation** – (domyślnie off) włącza mechanizm walidacji modelu w
 trakcie treningu. Walidacja polega na okresowym sprawdzaniu straty (loss) na osobnym
 zbiorze walidacyjnym, aby wykryć nadmierne dopasowanie (overfitting). 
 Gdy włączysz tę opcję, w zakładce *Concepts* będziesz mógł dodać
 specjalny koncept walidacyjny (zbiory obrazów nieuczestniczących w
 treningu) i ustalić co ile kroków/epok przeprowadzać walidację
 (**Validate** **after** – wartość liczbowa oraz jednostka: np. 1000 steps lub 1 epoch). Wyniki walidacji obejrzysz w TensorBoard jako osobny wykres straty
 dla zbioru walidacyjnego.

 • **Train** **device** – (domyślnie cuda ) określa urządzenie do
 trenowania. W praktyce wpisujesz tu np. cuda (dla domyślnej karty GPU)
 lub cuda:1 (jeśli masz wiele GPU i chcesz użyć drugiego). OneTrainer nie obsługuje jednoczesnego multi-GPU, ale można wskazać
 konkretną kartę. Ustawienie cpu spowoduje trenowanie na procesorze
 (bardzo wolne – tylko do testów).

 • **Temp** **device** – (domyślnie cpu ) określa urządzenie
 tymczasowe, na którym przechowywane będą elementy modelu, gdy nie są
 aktualnie używane. Domyślnie jest to CPU (pamięć RAM), co oznacza, że OneTrainer
 automatycznie *ofloaduje* (przenosi) część modelu do RAM, by
 oszczędzać VRAM karty graficznej. Pozwala to trenować większe modele
 na kartach o mniejszej pamięci, kosztem obciążenia RAM i CPU. Jeśli
 chcesz *wyłączyć* *ofloading*, możesz ustawić **Temp** **device**
 **=** **cuda** (wtedy wszystko trzymane będzie w VRAM). Więcej o ofloadingu w sekcji Zaawansowane opcje.

*(Screenshot:* *Przykładowa* *konfiguracja* *zakładki* *General* *z*
*ustawionymi* *ścieżkami* *i* *włączonym* *TensorBoard.)*

## Zakładka Model

W zakładce **Model** podajesz parametry dotyczące modelu bazowego oraz
formatu zapisu rezultatów. Innymi słowy, tu wskazujesz *na* *czym*
*trenujemy* i *jak* *zapisać* *wytrenowany* *model*. Opcje dostępne w
tej zakładce to:

 • **Hugging** **Face** **Token:** pole na wpisanie *Tokenu* *API*
 *HuggingFace*. Nie jest to obowiązkowe dla modeli publicznych, ale
 jeśli chcesz pobierać modele oznaczone jako *gated* (np. SDXL, Stable
 Diffusion 3.0, Flux) bezpośrednio przez OneTrainer, musisz posiadać
 token (ze swojego konta HF) . Token zostanie zapisany lokalnie (w pliku secrets.json ) i użyty
 automatycznie przy wczytywaniu modeli z HF. *Przykład:* model
 Flux.1-dev jest gated – mając token wpisany tutaj, OneTrainer może go
 pobrać z HuggingFace bez ręcznej interwencji.

 • **Base** **Model:** wskazanie modelu bazowego do trenowania.
 Domyślnie pole to zawiera link do modelu z HuggingFace (może to być
 np. Stable Diffusion XL 1.0 jeśli wybrałeś preset SDXL) . Możesz tu wpisać:

 • **Ścieżkę** **do** **pliku** **.safetensors/.ckpt** (np. lokalny
 checkpoint SD 1.5) lub

 • **Ścieżkę** **do** **katalogu** **z** **modelu** **Diffusers** (dla
 modeli w formacie diffusers, które są katalogami z plikami
 model_index.json , unet , vae , text_encoder itp.) lub

 • **URL/ID** **modelu** **na** **HuggingFace** (np.
 stabilityai/stable-diffusion-2-1 albo link do konkretnego repo HF).

 OneTrainer obsługuje zarówno formaty checkpoint (ckpt/safetensors) jak
 i Diffusers – nie musisz nic konwertować. Jeśli np. chcesz trenować LoRA na SDXL, wskaż tutaj ścieżkę do
 **SDXL** **base** **model** (w formacie diffusers lub AIO
 safetensors). Dla modelu Flux musisz tu podać katalog z modelem Flux.1
 (jeśli go pobrałeś ręcznie z HF – patrz sekcja Flux).

 • **VAE** **Override:** opcjonalnie możesz podać ścieżkę do
 alternatywnego modelu VAE, który ma zostać użyty podczas treningu (i
 generowania próbek) . Jeśli zostawisz puste, używany będzie domyślny VAE z modelu
 bazowego. Override bywa przydatny, np. gdy chcesz użyć ulepszonego VAE
 dla SD1.5 (klasycznego) albo gdy trenujesz styl i chcesz określony
 VAE.

 • **Model** **Output** **Destination:** docelowa nazwa/ścieżka dla
 zapisywanego modelu wynikowego (twojego wytrenowanego LoRA lub
 checkpointu). Możesz podać pełną nazwę pliku (np.

 models/moj_lora.safetensors – wtedy plik z LoRA znajdzie się tam po
 treningu). Jeśli podasz istniejący folder, OneTrainer wygeneruje nazwę
 pliku automatycznie, używając prefixu kopii zapasowej i sygnatury
 czasu. **Uwaga:** Dla LoRA nazwa powinna mieć rozszerzenie .safetensors lub .ckpt (choć zalecane jest safetensors ze względów
 bezpieczeństwa).

 • **Output** **Format:** format zapisu – do wyboru safetensors
 (domyślnie) lub ckpt . Safetensors są preferowane (bezpieczniejsze, mniejsza szansa
 uszkodzenia modelu).

 • **Data** **Types:** parametry precyzji treningu. OneTrainer
 umożliwia wybór formatu danych dla wag i gradientów, co wpływa na
 zużycie VRAM i szybkość. Jeśli korzystasz z wbudowanych presetów,
 wartości te są już dobrane optymalnie . Zaawansowani użytkownicy mogą tu zmieniać m.in.:

 • Typ wag modelu (float32, float16, bfloat16, int8 NP4/NF4 itp.), •
 Typ wyjściowego modelu (np. LoRA w 16-bit czy 8-bit).

 **Uwaga:** Należy zmieniać te ustawienia tylko jeśli wiesz co robisz –
 domyślne są zwykle odpowiednie. Przykładowo, dla dużych modeli jak Flux rekomendowane jest użycie co
 najmniej **FP8** precyzji, by uniknąć artefaktów (Flux w niższej
 precyzji NF4 może dawać siatkowy wzór na obrazach).

*(Screenshot:* *Zakładka* *Model* *z* *wczytanym* *modelem* *bazowym*
*SDXL* *i* *ustawioną* *nazwą* *pliku* *wyjściowego* *LoRA.)*

## Zakładka Data (Dane)

W zakładce **Data** ustawiamy opcje dotyczące przetwarzania danych
treningowych, głównie związane z przyspieszaniem treningu i obsługą
różnych rozmiarów obrazów. Znajdują się tu przede wszystkim trzy
przełączniki:

 • **Aspect** **Ratio** **Bucketing:** (czyli *bucketing* *proporcji*
 *obrazu*) – **must-have** podczas treningu na obrazach o różnych
 proporcjach. Po włączeniu tej opcji OneTrainer automatycznie pogrupuje
 (przypisze do “bucketów”) obrazy o podobnych proporcjach i dopasuje
 ich rozmiary tak, by miały zbliżoną liczbę pikseli. Dzięki temu można jednocześnie trenować na obrazkach 1:1, 16:9, 3:2
 itd., bez rozciągania – każdy bucket ma swoją “bazową” rozdzielczość.
 To potężna funkcja: umożliwia trenowanie zróżnicowanych zbiorów bez
 ręcznego przycinania wszystkich do jednego formatu. **Zaleca** **się**
 **mieć** **włączone** (domyślnie włączone w presetach).

 • **Latent** **caching:** (*keszowanie* *latentów*) – po włączeniu
 OneTrainer będzie zapisywał w folderze cache wstępnie przetworzone
 wersje obrazów (ich reprezentacje latentne). Ponieważ generowanie latentów z obrazów (przepuszczanie przez
 encoder VAE) to operacja kosztowna, keszowanie przyspiesza kolejne
 epoki treningu – zamiast liczyć to za każdym razem, model wczyta z
 dysku wcześniej obliczone wartości. *Uwaga:* Kesz zajmuje miejsce na
 dysku (w zależności od datasetu może to być kilka-kilkanaście GB).
 Warto włączyć, jeśli masz dość miejsca – znacznie przyspiesza trening,
 zwłaszcza przy wielu epokach.

 • **Clear** **cache** **before** **training:** – czyszczenie cache
 przed startem treningu
 
 . Domyślnie **włączone**, co oznacza, że przy każdym nowym
 uruchomieniu treningu OneTrainer usunie stare pliki cache (z
 poprzednich konfiguracji/ustawień) i zbuduje je od nowa. Dzięki temu
 masz pewność, że zmiany w obrazach czy parametrach zostaną
 uwzględnione. Wyłącz tę opcję tylko, jeżeli **kontynuujesz** **ten**
 **sam** **trening** i nie zmieniasz nic w danych – inaczej stary cache
 może powodować błędy (np. jeśli zmieniono rozdzielczość docelową,
 stary cache nie pasuje i nastąpi błąd).

Dla początkujących zaleca się zostawić wszystkie powyższe opcje
**włączone** – co OneTrainer robi domyślnie w presetach. Dzięki nim trening będzie efektywniejszy i bardziej stabilny.

Ponadto w zakładce Data (w nowszych wersjach) może pojawić się opcja
**Dataloader** **Threads** – liczba wątków do wczytywania danych. Jeśli
posiadasz mocny CPU, możesz zwiększyć dla szybszego ładowania batchy
(np. posiadacze RTX 4090 często ustawiają 8 wątków). Ale ostrożnie: zbyt wiele wątków może zająć dodatkowy VRAM, zaleca się
testować (domyślnie chyba 1 lub 2).

*(Więcej* *informacji* *o* *bucketingu* *i* *proporcjach* *znajdziesz*
*w* *sekcji* *Proporcje obrazu i bucketing* *oraz* *na* *wiki*
[*56*]
*.)*

## Zakładka Concepts (Zbiór danych)

To kluczowa zakładka, gdzie konfigurujesz **zbiór** **treningowy**. W
OneTrainer pojęcie *Concepts* oznacza po prostu zestawy obrazów ze
swoimi podpisami, używane do treningu modeli. Możesz mieć wiele
konceptów – np. jeden główny (ze zdjęciami obiektu/osoby, których model
ma się nauczyć), drugi z obrazami regularyzującymi (tzw. negative
class/prior), trzeci jako walidacja itd.
Zakładka Concepts umożliwia dodawanie, usuwanie i konfigurowanie
takich podzbiorów.

Główne elementy UI w tej zakładce to:

 • **Dropdown** **Menu** **(Concept** **Config):** lista konfiguracji
 konceptów. Domyślnie jest jedna konfiguracja *concepts*, ale możesz
 dodać kolejne. OneTrainer trenuje tylko tę konfigurację, która jest
 aktualnie wybrana na liście. (Funkcja ta bywa używana, by szybko przełączać się między różnymi
 zestawami danych/treningami bez ręcznego przeładowywania wszystkiego).

 • **Add** **Config** **/** **Delete** **Config:** przyciski do
 tworzenia lub usunięcia całej konfiguracji konceptów (czyli zestawu
 konceptów).

 • **Add** **Concept:** przycisk dodający nowy *concept* do aktualnej
 konfiguracji. Gdy go klikniesz, pojawi się nowa pozycja na liście z domyślną nazwą
 – kliknij ją, aby otworzyć okno edycji szczegółów konceptu
 (alternatywnie kliknij ikonę *Edit*).

 • **Lista** **Conceptów:** poniżej, każdy concept jest wyświetlany (z
 nazwą lub ścieżką). Możesz włączać/ wyłączać koncept (toggle
 **Enable**, niebieski gdy włączony) – co wpływa na to, czy będzie
 brany pod uwagę w treningu. Są też ikony: czerwony X usuwa concept, zielony + duplikuje go (kopiuje ustawienia i pozwala np. podmienić
 ścieżkę).

Pododaniu/wybraniuconceptukliknięciegootworzy**oknoustawieńconceptu**,podzielonenazakładki.
Tam konfigurujemy szczegółowo dany zbiór danych. Najważniejsze
ustawienia (zakładka *General* w oknie conceptu) to:

 • **Name:** Nazwa konceptu (możesz nazwać np. „moja_postac” lub
 zostawić puste – wtedy nazwa zostanie nadana na podstawie nazwy
 folderu po zatwierdzeniu).

 • **Enabled:** to samo co toggle na liście – czy concept jest aktywny
 (domyślnie True) .

 • **Concept** **Type:** typ konceptu – *Standard* (zwykłe dane
 treningowe), *Validation* (dane tylko do walidacji – zostaną użyte
 przy obliczaniu straty walidacyjnej, ale nie wpłyną na trening) albo
 *Prior* (tzw. prior preservation – używane w DreamBooth do
 zapobiegania nadpisaniu oryginalnych cech modelu przez generowanie
 obrazów „klasy”). Jeśli twój concept ma pełnić rolę obrazów regularyzujących (tzw.
 „negative class images”), wybierz **Prior**. Dla walidacyjnego –
 **Validation** (pamiętaj dodać ich włączenie w zakładce General →
 Validation). Standard to zwykły treningowy zbiór.

 • **Path:** ścieżka do folderu z obrazami dla tego conceptu. Tutaj wskazujesz katalog, w którym znajdują się pliki obrazów
 (JPEG/PNG) i ewentualnie ich podpisy .txt. Po wybraniu folderu
 OneTrainer automatycznie wczyta listę plików. **Uwaga:** Zadbaj o
 poprawną strukturę – najlepszą praktyką jest trzymanie obrazów i
 plików .txt parami o tej samej nazwie (np. 001.jpg i 001.txt ). Jeśli pliki .txt są w osobnym folderze lub inaczej nazwane
 – dopasuj to opcjami *Prompt* *Source*.

 • **Prompt** **Source:** sposób wczytywania podpisów do obrazów. Możliwe opcje:

 • **From** **text** **file** **per** **sample** – domyślnie. Każdemu
 obrazowi odpowiada plik .txt o tej samej nazwie (np. obraz 001.jpg i
 tekst 001.txt ). Jeśli w pliku jest wiele linijek, OneTrainer będzie losowo wybierał jedną linijkę jako podpis w danej epoce, co
 pozwala mieć wiele wariantów opisu dla jednego obrazu.

 • **From** **single** **text** **file** – wskazujesz jeden plik .txt,
 którego zawartość będzie używana jako ten sam podpis dla wszystkich
 obrazów . Rzadziej używane (może przy stylach).

 • **From** **image** **file** **name** – OneTrainer odczyta podpis z
 nazwy pliku obrazu (np. sunset_beach.png → podpis „sunset beach”). Użyteczne, jeśli Twoje pliki są nazwane tagami zamiast mieć osobne .txt.

 • **Include** **Subdirectories:** (domyślnie False) – jeśli włączysz,
 OneTrainer rekurencyjnie weźmie obrazy również z podfolderów wskazanej
 ścieżki. Przydatne, gdy masz dane podzielone w podkatalogach, a chcesz
 traktować je jako jeden concept.

 • **Image** **Variations:** (domyślnie 1) – parametry zaawansowane,
 związane z augmentation i caching. Określa ile *wariantów* *obrazów*
 będzie keszowanych z uwzględnieniem augmentacji. Jeśli używasz augmentacji losowych (patrz niżej) oraz latent
 caching, możesz zwiększyć tę liczbę, by przechować kilka
 zróżnicowanych wersji każdego obrazu. Standardowo 1 wystarcza (przy
 włączonym caching i augmentacjach musisz ustawić *Image* *Variations*
 *=* *1*).

 • **Text** **Variations:** (domyślnie 1) – analogicznie, ile
 *wariantów* *tekstu* (podpisu) będzie keszowanych na obraz
 
 . Jeśli nie trenujesz encoderów tekstowych i korzystasz z caching,
 zaleca się zwiększyć tę wartość, aby uniknąć sytuacji, że jeden
 wylosowany podpis zostanie użyty w każdej epoce (z cache). Generalnie,
 jeśli masz wiele epok i wiele możliwych podpisów na obraz, daj większe
 *Text* *Variations* by rotacja była większa.

 • **Balancing:** (domyślnie *Repeats:* *1*) – niezwykle ważna opcja,
 gdy masz nierówne ilości obrazów w konceptach. Balancing pozwala
 zbalansować wkład każdego conceptu w trening. Dwa tryby:

 • **Repeats:** Podajesz liczbę powtórzeń – oznacza to, że każdy obraz
 z tego conceptu będzie liczył się jak *n* obrazów. Np. jeśli concept A
 ma 100 obrazów, concept B ma 1000 obrazów, możesz ustawić repeats=10
 dla A, by efektywnie zrównać ich wpływ. Domyślne 1 oznacza brak
 powtórzeń.

 • **Samples:** Alternatywny tryb – możesz zamiast powtarzać, określić
 **dokładną** **liczbę** **obrazów** **z** **tego** **conceptu** **na**
 **epokę** (będzie losowo wybierana ta liczba spośród wszystkich). Ten
 tryb bywa użyteczny przy bardzo dużych zbiorach regularyzacyjnych –
 możesz np. ustawić, że w każdej epoce weź 200 losowych
 regularyzacyjnych spośród 10k dostępnych, zamiast używać wszystkich.

 • **Loss** **Weight:** (domyślnie 1.0) – waga straty dla tego conceptu. Pozwala zmniejszyć wpływ któregoś zbioru na dostrajanie. Np. jeśli
 używasz obrazów regularyzacyjnych (prior preservation) i zauważasz, że
 zbyt mocno one wpływają hamująco na trening, możesz dać im wagę np.
 0.5 – wówczas gradient z nich będzie mnożony przez 0.5, czyli ich
 wpływ na parametry będzie mniejszy. Zazwyczaj zostaw 1, chyba że wiesz
 co robisz.

Po skonfigurowaniu powyższych ustawień kliknij **OK/Zatwierdź** (w GUI
to zwykle zamknięcie okna conceptu zapisuje zmiany). **90%** **sukcesu**
**treningu** **zależy** **od** **jakości** **i** **różnorodności**
**danych** **oraz** **dobrych** **podpisów** – poświęć czas na
przygotowanie datasetu, bo to najważniejsza część procesu. Zaleca się zgromadzić **wysokiej** **jakości,** **zróżnicowane**
**obrazy** oraz stworzyć do nich **dokładne** **i** **różnorodne**
**opisy** **(captions)** – to klucz do dobrego LoRA.

*(Screenshot:* *Okno* *konfiguracji* *Concept* *–* *ścieżka* *do*
*obrazów,* *typ* *„Standard”,* *źródło* *podpisów* *z* *plików* *.txt,*
*repeats* *ustawione* *tak,* *by* *zbalansować* *mały* *zbiór* *wobec*
*dużego* *regularization.)*


**Zakładki** **dodatkowe** **w** **oknie** **Concept:** Poza zakładką
główną (General), zobaczysz też zakładkę **Image** **Augmentation** i
ewentualnie **Advanced**. Omówmy krótko augmentation, bo to integralna
część przygotowania danych:

 • **Image** **Augmentation:** tutaj możesz ustawić automatyczne
 augmentacje obrazów, które zwiększą różnorodność danych. Augmentacje są *szczególnie* *ważne* *przy* *małych* *datasetach*,
 by model nie naduczył się konkretnych obrazków. Dostępne opcje:

 • **Crop** **Jitter:** (domyślnie On) – jeśli obraz wymaga przycięcia
 do docelowego wymiaru, włączony crop jitter przesuwa losowo kadr (nie
 zawsze centralnie), co dodaje odmianę w kadrowaniu.

 • **Random** **Flip:** (On) – losowe odbicie lustrzane obrazu w
 poziomie. Typowa augmentacja symetryczna – np. zdjęcie lewoskrętne vs
 prawoskrętne.

 • **Random** **Rotation:** (Off domyślnie, wartość 0) – losowa rotacja
 obrazu w pewnym zakresie stopni. Możesz ustawić np. do 10° losowej
 rotacji, by urozmaicić perspektywę.

 • **Random** **Brightness** **/** **Contrast** **/** **Saturation**
 **/** **Hue:** (domyślnie Off, wartości 0) – te ustawienia pozwalają
 losowo rozjaśniać/przyciemniać obraz, zmieniać kontrast, nasycenie
 kolorów i odcień. Podajesz zakres (np. ±0.1) lub wartość stałą. Dzięki nim model uczy
 się pewnej odporności na zmiany oświetlenia, kolorystyki itd. Dobrze
 jest włączyć drobne losowe zmiany jasności/ kontrastu dla lepszej
 ogólności. OneTrainer umożliwia też ustawienie tych zmian jako stałe
 (fixed) zamiast random – wtedy co epokę to samo przekształcenie.
 Zwykle jednak dajemy losowe.

 • **Update** **Preview:** przycisk generujący podgląd augmentacji –
 pozwala zobaczyć jak wygląda przykładowy obraz po zastosowaniu
 aktualnie ustawionych transformacji. Dla ustawień losowych można klikać kilka razy, by zobaczyć różne
 warianty.

Augmentacje można stosować **losowo** (random) lub **deterministycznie**
(stały offset) – OneTrainer daje wybór. Należy pamiętać, że jeśli
włączamy augmentacje losowe *i* używamy latent cache, to aby każda epoka
mogła mieć inne warianty, musimy albo wyłączyć caching co epokę, albo
ustawić parametry *Image/Text* *Variations* ** *1*. To już
zaawansowana kwestia – generalnie korzystaj z augmentacji z umiarem i
tylko gdy dataset jest mały lub jednorodny.

*(Więcej* *praktycznych* *porad* *dot.* *przygotowania* *datasetu* *–*
*patrz* *Praktyczne porady.* *Możesz* *też* *skorzystać* *z*
*narzędzi* *automatycznego* *tagowania* *i* *maskowania* *w* *zakładce*
*Tools* *zamiast* *ręcznie* *pisać* *opisy,* *o* *czym* *w* *dalszej*
*części.)*

## Zakładka Training

W zakładce **Training** ustawiamy hiperparametry treningu: m.in.
liczebność batchy, ilość epok, learning rate, wybór optymalizatora,
scheduler itp. Jest to najbardziej „techniczna” zakładka, której
opanowanie wymaga zrozumienia ML. Na szczęście, **presety**
**OneTrainer** (wybrane w menu konfig na górze) zawierają zazwyczaj
rozsądne domyślne wartości – początkującym poleca się ich trzymać. Tutaj
opiszemy główne parametry:

*(Uwaga:* *Interfejs* *mógł* *ulec* *zmianie* *i* *zawierać* *więcej*
*sekcji;* *OneTrainer* *jest* *intensywnie* *rozwijany.* *Poniższy*
*opis* *bazuje* *na* *dokumentacji* *z* *maja* *2025* *–* *wszelkie*
*parametry* *mają* *dymki* *pomocy* *w* *GUI,* *co* *ułatwia*
*zrozumienie*
[*84*]
*.)*

 • **Learning** **Rate** **(LR):** podstawowy współczynnik uczenia.
 Często rozdzielony na LR dla UNet i LR dla Text Encodera:

 • **Base** **LR** **(UNet** **LR):** szybkość uczenia parametrów UNet
 (głównej sieci generatywnej).

 • **Text** **Encoder** **LR:** szybkość uczenia parametrów enkodera
 tekstowego. W SDXL są dwa encodery (CLIP ViT-L i OpenCLIP ViT-G) –
 OneTrainer umożliwia sterowanie dwoma LR dla TEnc1 i TEnc2. Zwykle LR dla tekstu ustawia się niżej niż UNet (albo w ogóle
 zamraża encodery tekstowe na początku, zależnie od strategii).

 • Warto wspomnieć: w SDXL panuje opinia, że pierwszy encoder lepiej
 pracuje z tagami, drugi z pełnymi opisami – ale to nie twarda reguła. Trenowanie tekst encodera jest trudne i często pomijane przy LoRA,
 ale OneTrainer daje taką opcję.

 • **Optimizer** **i** **Scheduler:** wybór optymalizatora (AdamW,
 Lion, AdaFactor, DAdaptation itp.) i harmonogramu LR (cosine, linear,
 constant etc.). OneTrainer stale dodaje nowe optymalizatory – jest
 nawet osobna strona wiki o nich. Domyślny dla LoRA bywa AdamW lub Prodigy/ dAdaptation (które
 automatycznie dostosowują LR). Jeśli nie wiesz, zostaw jak w presecie.
 Scheduler decyduje jak LR spada w trakcie treningu.

 • **Batch** **Size** **i** **Gradient** **Accumulation:** ile obrazów
 idzie na raz (batch) i czy gradient jest akumulowany przez kilka
 batchy zanim nastąpi krok optymalizatora. Jeśli masz mocną kartę,
 możesz zwiększyć batch size dla stabilniejszego treningu. Gradient
 Accumulation pozwala efektywnie uzyskać duży *effective* *batch*
 dzieląc go na mniejsze paczki – np. batch 2 z accum 4 daje efektywnie
 batch 8 (2*4). Ustawienia te zależą od VRAM – w razie błędów OOM
 zmniejsz batch.

 • **Epochs** **/** **Steps:** możesz ustawić liczbę *epochs* (przejść
 przez cały zbiór) lub konkretną liczbę *steps* treningowych.
 OneTrainer pozwala też ustawić limit czasu treningu. Presety zwykle
 definiują np. 10 epoch przy repeats=..., co skutkuje ~ określoną
 liczbą kroków.

 • **Save** **frequency** **(Backup):** co ile kroków/epok zapisywać
 kopie zapasowe modelu. OneTrainer automatycznie wykonuje regularne
 backupy, zapisywane w workspace/<run/checkpoints z prefixem i
 timestamp. W zakładce Training (lub Backup) możesz ustawić np. *save* *every*
 *1000* *steps* albo *every* *epoch*. Dzięki temu nie stracisz postępów
 jak coś przerwie trening. (Jest też opcja *keep* *only* *last* *N*
 *backups* aby nie zapełnić dysku).

 • **EMA** **(Exponential** **Moving** **Average):** OneTrainer
 obsługuje też utrzymanie średniej wykładniczej wag podczas treningu. EMA potrafi poprawić jakość modelu generatywnego. Włączenie EMA
 zwiększy jednak zużycie VRAM (OneTrainer ma opcję trzymania EMA wag na
 CPU by zmniejszyć VRAM użycie). Opcja ta może być włączana w Training
 tab. Przy SD1.5 często nie używana, dla SDXL potrafi pomóc (ale
 wydłuża czasy).

 • **Masked** **training:** (szkolenie z maskami) – OneTrainer wspiera
 trenowanie modeli tylko na określonych fragmentach obrazów. W Training
 zakładce możesz włączyć *Use* *Masks* i ustawić parametry:

 • **Unmasked** **probability** (domyślnie 0.1) – ułamek kroków
 treningu, w których *nie* używa się maski (czyli cały obraz jest
 trenowany). Np. 0.1 oznacza, że 10% kroków ignoruje maskę (dla stabilizacji
 treningu).

 • **Unmasked** **weight** (domyślnie 0.1) – jeśli maska jest wyłączona
 na danym kroku, możesz tu ustawić jak bardzo te kroki wpływają (waga
 strat z obszaru maski).

 • **Normalize** **Masked** **Area** **Loss** – opcja normalizacji
 straty dla obszaru maski
 
 . Zaleca się *włączać,* *gdy* *maska* *zajmuje* *duży* *obszar*
 *obrazu* (np. maseczka na całej twarzy), aby strata była prawidłowo
 skalowana; dla małych masek nie włączaj (może zwiększyć loss
 niepotrzebnie).

Maski to czarno-białe obrazy towarzyszące obrazom treningowym, gdzie
białe obszary oznaczają region ważny (model ma *skupić* *się* na nim), a
czarne – mniej istotny. Pliki masek muszą mieć taką samą nazwę co obraz + sufix -masklabel.png
i być w formacie PNG. OneTrainer w zakładce Tools potrafi automatycznie generować maski (o
czym niżej). Maski przydają się np. gdy trenujesz LoRA na konkretny
obiekt na obrazach i chcesz, by model mniej uczył się tła. - **Validation** **settings:** Jeśli włączyłeś walidację w General
tab, to w Training możesz ustawić np. **Validation** **interval** (co
ile kroków/ epok liczyć walidację, domyślnie to samo co sample interval
zazwyczaj). Walidacja działa tak, że w trakcie treningu co zadany interwał model
sprawdza się na obrazach walidacyjnych (Concept Type: Validation) i
liczy średni loss – obserwując wykres tego lossu w TensorBoard możesz
wykryć moment, gdy model zaczyna przeuczać (loss walidacyjny rośnie, podczas gdy
treningowy spada). Wtedy warto przerwać trening lub zastosować early stopping.

OneTrainer posiada *tooltips* *(dymki* *pomocy)* dla każdej opcji – nie
bój się najechać kursorem, by zobaczyć wyjaśnienia w UI. Jeśli dopiero zaczynasz, **nie** **zmieniaj** **zbyt** **wiele** –
użyj ustawień z presetu. Twórcy OneTrainer zaznaczają, że interfejs jest
stale ulepszany, więc screenshoty w wiki mogą być nieaktualne – ufaj
opisom/dymkom.

*(Screenshot:* *Zakładka* *Training* *z* *ustawieniami:* *batch* *size*
*2,* *10* *epok,* *optimizer* *AdamW,* *LR* *1e-4,* *mask* *training*
*on* *z* *parametrami.)*

**Zakładka** **LoRA**

Ta zakładka pojawia się **tylko** **jeśli** **w** **ogólnych**
**ustawieniach** **wybrałeś** **tryb** **LoRA** (górny pasek, dropdown
obok logo OneTrainer: Training Mode = LoRA ). Dotyczy ona specyficznych parametrów trenowania LoRA – czyli
niskowymiarowych adapterów. Omówmy dostępne opcje:

 • **Type:** rodzaj algorytmu LoRA – do wyboru **LoRA** **(klasyczna)**
 lub **LoHa**. *LoHa* (Low-rank adaptation with Hadamard product) to wariant
 wprowadzony przez projekt LyCORIS, który potrafi uchwycić więcej
 informacji niż klasyczna LoRA kosztem większej złożoności. Jeśli chcesz trenować LoHa, wybierz tę opcję. (OneTrainer od
 kwietnia 2024 obsługuje LoHa tak samo jak LoRA.)

 • **LoRA** **base** **model:** pozwala wczytać istniejącą LoRA jako
 punkt startowy do dalszego trenowania (resume). Jeśli np. masz LoRA, którą chcesz podtrenować, możesz wskazać plik
 .safetensors tutaj. Alternatywnie można też wznowić z *katalogu*
 *backupu* *treningu* (OneTrainer backup), ale wtedy pewne rzeczy nie
 mogą być zmienione (np. architektura). Zwykle to pole zostawiasz
 puste, chyba że wznawiasz trening LoRA.

 • **LoRA** **rank:** domyślnie 16 (dla SD1.5), zalecane 8 lub 16 dla
 SDXL. Parametr ten określa „rzutowanie” wag – im wyższy rank, tym więcej
 parametrów w LoRA i potencjalnie większa jego pojemność, ale też
 łatwiej o przeuczenie i większe zużycie VRAM. Praktycznie:

 • Dla modeli 1.5 standardowo rank 16 to dobry wybór.

 • Dla SDXL niektórzy stosują rank 8, bo SDXL ma ogromne UNet i LoRA
 rank 16 to już bardzo dużo parametrów (stąd podatność na przeuczenie).
 
 • Wyższe ranki (16) zazwyczaj nie poprawiają jakości, a mogą ją
 pogorszyć (przeuczenie), chyba że masz *bardzo* zróżnicowany i duży
 dataset.

 • **LoRA** **alpha:** domyślnie 1.0. To hiperparametr skalujący – w
 implementacji LoRA końcowe wagi są mnożone przez (alpha/rank). Jeśli zostawisz 1.0 przy rank 16, to efektywnie mnożnik LR wynosi
 1/16 = 0.0625. Często tak jest OK. Jeżeli zmieniasz alpha, pamiętaj by odpowiednio
 dostosować learning rate (bo alpha wpływa na siłę uczenia). Ogólna zasada: **nie** **musisz** **zmieniać** **alpha**, chyba że
 wiesz po co. Niektórzy zmniejszają alpha przy wysokich rankach lub
 przy użyciu określonych optymalizatorów, by “osłabić” uczenie LoRA.

 • **Decompose** **Weights:** (DoRA) – *mocno* *zaawansowana* *opcja.*
 Po włączeniu OneTrainer zastosuje dekompozycję wag LoRA na
 **magnitudę** **i** **kierunek** (Magnitude/Direction decomposition). Jest to technika znana jako **DoRA** (Weight Decomposition for LoRA)
 – polega na rozbiciu uczenia na dwa składowe tory. Badania pokazują,
 że DoRA potrafi **znacznie** **poprawić** **jakość** **uczenia** i
 szybciej konwergować, często dorównując pełnemu fine-tuningowi modeli. W praktyce, jeśli włączysz tę opcję, **koniecznie** **zmniejsz**
 **dropout** (nawet do 10x mniejszego niż normalnie), bo DoRA silniej oddziałuje na parametry. DoRA to świetna opcja dla
 zaawansowanych – warto spróbować, jeśli normalna LoRA nie daje
 satysfakcjonujących wyników, ale wymaga to dopracowania innych
 hiperparametrów.
 
 • **Dropout** **probability:** (domyślnie 0.0) – prawdopodobieństwo
 dropout w warstwach LoRA. Dropout to technika regularyzacji – losowo dezaktywuje część
 neuronów w trakcie treningu, by model się nie przyzwyczajał. Jeśli
 obserwujesz przeuczenie LoRA, rozważ wprowadzenie dropout np. 0.1–0.3.
 Zwłaszcza przy DoRA, jak wyżej wspomniano, dropout 0.1–0.5 bywa
 zalecany. Bez DoRA zazwyczaj LoRA 0.1 dropout jest bezpieczne podejście by
 poprawić generalizację.

 • **LoRA** **weight** **data** **type:** typ danych dla wag LoRA w
 trakcie wczytywania. Domyślnie float32 (pełna precyzja). Można zmniejszyć np. do float16,
 żeby oszczędzić VRAM podczas **wczytywania** LoRA. Nie wpływa to na
 końcowy zapis – to tylko kwestia pamięci operacyjnej. Raczej nie
 ruszaj, jeśli VRAM nie jest krytyczny.

 • **Bundle** **Embeddings:** (domyślnie ON) – opcja dołączania
 ewentualnie trenowanych jednocześnie embeddingów (tekstowych) do pliku
 LoRA. OneTrainer pozwala trenować jednocześnie LoRA i textual inversion
 (embedding) – jeśli to robisz i ta opcja jest ON, to wynikowy plik
 LoRA będzie zawierał także wytrenowane embeddingi. Taki „spakowany”
 plik działa w Auto1111 i SD.Next bez problemu (czytają one te
 dodatkowe tensorzyki), natomiast np. ComfyUI tego nie obsłuży. Jeśli trenujesz tylko LoRA, ta opcja nic nie zmienia (ale nie
 szkodzi, może zostać włączona).

 • **Layer** **Preset:** wybór, które warstwy modelu będą objęte LoRA. Historycznie LoRA trenowała tylko *bloki* *uwagi* *(attention*
 *layers)* modelu Stable Diffusion. W OneTrainer teraz możesz wybierać:

 • **Attention** – tylko warstwy typu Attention (najbezpieczniej,
 mniejsza szansa artefaktów).

 • **Attention** **+** **MLP** – warstwy Attention oraz dodatkowo
 warstwy pełne (feed-forward, MLP). To ustawienie domyślne w narzędziach Kohya.

 • **All** – LoRA na wszystkich warstwach modelu (U-Net), co zbliża się
 do pełnego fine-tune – zazwyczaj nie zalecane bez powodu.

 • **Custom** – możesz ręcznie wpisać listę warstw do trenowania
 (zaawansowane; OneTrainer pozwala np. wpisać indeksy bloków/warstw do
 objęcia LoRA).
 
Warstwy wybierzesz z listy presetów lub własnoręcznie – **zaleca**
**się** **użyć** **gotowych** **(attention** **lub** **attn+MLP)**,
chyba że wiesz, które parametry chcesz edytować. Dodanie MLP warstw może
pomóc LoRA
nauczyćsięwięcejaspektów,aleteżzwiększaryzyko,żeLoRAbędziemniejprzenośnamiędzymodelami
i może np. powodować nasycenie kolorów (MLP kontrolują pewne
statystyki). Jednak wiele LoR dostrojonych do stylów używa attn+MLP dla
lepszego efektu. Możliwość wyboru warstw to przydatna nowość OneTrainer,
zwiększająca elastyczność treningu.

*(Screenshot:* *Zakładka* *LoRA* *–* *tryb* *LoRA,* *typ* *LoRA*
*wybrany* *na* *LoHa,* *rank* *8,* *alpha* *1,* *dropout* *0.1,* *DoRA*
*włączone,* *layers* *=* *attn+MLP.)*

**Zakładka** **Sampling** **i** **Backup**

OneTrainer posiada również zakładki pozwalające na generowanie
podglądowych obrazów podczas treningu (**Sampling**) oraz zarządzanie
kopiami zapasowymi/wynikami (**Backup/Save**). W niektórych wersjach GUI
mogą to być osobne zakładki, w innych razem w sekcji Training/Sampling.
Opisujemy je razem, bo są związane z monitorowaniem postępów:

 • **Sampling** **(Sample** **images):** Umożliwia ustawienie
 automatycznego generowania obrazów próbnych w trakcie treningu. Możesz
 określić:

 • Co ile kroków lub epok ma być wygenerowana próbka (**Sample**
 **after** **n** **steps/epochs**).

 • Liczbę próbek i ewentualne parametry generowania (domyślnie
 wykorzystuje generowanie wewnątrz OneTrainer – model podczas treningu
 produkuje obraz przy zadanym promptcie, byś mógł ocenić czy już
 nauczył się koncepcji).

 • Prompt(y) do próbek – często można wpisać kilka stałych promptów,
 które model ma generować przy kolejnych próbkach.

 • Dodatkowe opcje jak seed, czy zapisywać do osobnego folderu.

Generowanie próbek jest **opcjonalne,** **ale** **bardzo** **przydatne**
– możesz **wizualnie** **śledzić** **postępy** modelu. Jeśli dopiero zaczynasz, możesz nie wiedzieć jak interpretować takie
próbki, ale warto to robić – z czasem zobaczysz np. że obrazki zaczynają
wyglądać sensownie, a po pewnym czasie może zaczynają być zbyt „sztywne”
(sygnał do zakończenia treningu). Próbki zapisują się w workspace/ <run/samples/… i także są widoczne
w TensorBoard (jako obrazy w zakładce Images, o ile TensorBoard jest
włączony). *Pro* *tip:* Ustaw prompt próbek tak, by odzwierciedlał
docelowe użycie modelu (np. zawrzyj token/kontrolne słowo unikalne,
jeśli trenujesz konkretny obiekt). - **Backup/Save:** Ta część pozwala
ustawić szczegóły zapisu modeli: - Prefix dla nazw kopii zapasowych (np.
MyLora – wtedy pliki będą MyLora_epoch_1.safetensors itp.). - Czy
zapisywać tylko LoRA czy także pełny model (przy LoRA zwykle tylko
LoRA). - Ile kopii trzymać (przy długim treningu można trzymać np. tylko
2 ostatnie, by oszczędzić miejsce). - Ewentualnie czy automatycznie
zapisać finalny model po zakończeniu.

OneTrainer robi pełne backupy zawierające **wszystkie** **informacje**
**potrzebne** **do** **wznowienia** **treningu** (stan optymalizatora,
itp.). Dzięki temu nawet jeśli przerwiesz proces, możesz później wybrać
*Continue* *from* *last* *backup* w General i ruszyć dalej bez utraty
postępów. Kopie te są jednak spore (zawierają model, optymizer, EMA itp.).
Możesz ręcznie usuwać stare, gdy niepotrzebne.

Zarówno próbki jak i backupy są *opcjonalne* – można trenować bez nich.
Ale dla realnych projektów zaleca się korzystać z obu: backupy chronią
przed stratą czasu w razie awarii, a sample pomogą ocenić jakość modelu
w trakcie.

*(Screenshot:* *Zakładka* *Sampling* *–* *ustawione* *co* *1* *epoch*
*generuj* *2* *obrazki* *z* *zadanym* *promptem;* *Zakładka* *Backup*
*–* *prefix* *ustawiony,* *zachowaj* *ostatnie* *3* *kopie,* *zapisuj*
*co* *1* *epoch.)*

**Zakładka** **Tools** **(Narzędzia)**

Zakładka Tools zawiera **przydatne** **narzędzia** **dodatkowe**, które
upraszczają przygotowanie danych i rozszerzają możliwości OneTrainer. W
szczególności znajdują się tu:

 • **Dataset** **Tools** **(Caption** **&** **Mask):** interfejs do
 automatycznego generowania opisów (captioning) dla obrazów oraz masek
 dla masked training. Jeśli Twój zbiór obrazów nie ma jeszcze plików .txt z podpisami,
 możesz skorzystać z wbudowanych algorytmów:

 • *BLIP/BLIP2* – modele generujące opisy sceny (pełne zdania) na
 podstawie zawartości obrazu.

 • *WD14* *Tagger* – klasyfikator tagów (przeszkolony na Danbooru),
 generuje listę tagów opisujących obraz.

 Narzędzie pozwala wybrać model captionujący, dodać ewentualny
 prefix/sufix do generowanych podpisów (np. stałą frazę), a także skorzystać z istniejących wstępnych podpisów (jeśli np.
 część obrazów jest już opisana). Po konfiguracji możesz uruchomić
 generowanie – OneTrainer przejdzie po obrazach i stworzy dla nich
 pliki .txt. *Wskazówka:* WD14 wymaga czasem doinstalowania dodatkowego
 modelu; upewnij się, że masz połączenie internetowe lub pobierz model
 taggera.

Co do masek: Tools oferuje *Batch* *Mask* *Generation* – możesz
automatycznie wygenerować maski dla obrazów za pomocą: - **ClipSeg** –
model segmentacji z wykorzystaniem promptu tekstowego (np. wpisujesz „a
woman’s face”, a ClipSeg spróbuje zamaskować twarz kobiety na zdjęciu). - **Rembg** **/** **Rembg-human** – modele do usuwania tła
(separowania obiektu od tła). Świetne do generowania masek np. osób,
produktów itp. - **Hex** **Color** – narzędzie do maskowania według
zadanego koloru (np. wszystkie zielone piksele zamaskuj).

 13

Możesz wybrać narzędzie i uruchomić – OneTrainer wygeneruje pliki
*-masklabel.png dla każdego obrazu. Dostępne są też funkcje manualne:
malowanie własnej maski i użycie *fill* (wypełnienie reszty). Po zakończeniu edycji maski *pamiętaj* *nacisnąć* *Enter,* *żeby*
 *zapisać* *zmiany*
 
 .

**Dataset** **Tools** są niezwykle wygodne – dzięki nim cały pipeline
(tagowanie obrazów, maskowanie) można zrobić wewnątrz OneTrainer, bez
potrzeby używania osobnych skryptów. Jest to szczególnie przydatne dla
przygotowania DreamBooth (gdzie generujemy obrazy klasy/regularyzacyjne
i tagujemy je jednym słowem) czy innych LoRA (np. stylu – tagger WD14
szybko opisze style). - **Video** **Tools:** OneTrainer potrafi także
pomóc w pozyskiwaniu danych z wideo. Zakładka Tools ma sekcję do wyciągania klatek z filmów: - Możesz
wskazać plik wideo (lub folder z wieloma) i folder wyjściowy, ustawić
interwał co ile klatek/sekund wyciągnąć obraz. - Opcja “Output to
Subdirectories” pozwala tworzyć osobne podfoldery dla każdego filmu. - Narzędzie może też pociąć długie wideo na krótsze klipy przed
wyciąganiem klatek (zalecane ręcznie pociąć, bo one i tak lecą
równolegle).

**Uwaga:** do obsługi różnych formatów wideo może być wymagany ffmpeg –
jeśli video tools nie działa dla jakiegoś pliku, zainstaluj ffmpeg w
systemie.

Video tools są przydatne, gdy tworzysz dataset np. z nagrań (np. chcesz
model który generuje klatki z filmu – możesz wyciągnąć klatki i je
trenować). - **Model** **Tools** **(Convert):** (może być osobna
zakładka lub w Tools) – OneTrainer ma także narzędzia do konwersji
modeli między formatami. Np. możesz załadować checkpoint .ckpt i zapisać go jako diffusers,
albo połączyć LoRA z modelem bazowym (tzw. merge LoRA do checkpointa). W
UI jest to zwykle prosty kreator.

Ogółem zakładka Tools jest zbiorem „dodatków”, które czynią OneTrainer
prawdziwie **jednym** **narzędziem** **do** **wszystkiego** – od
przygotowania danych, przez trening, po konwersje modeli.

*(Screenshot:* *Zakładka* *Tools* *–* *zaznaczone* *opcje* *BLIP2* *i*
*WD14* *do* *tagowania* *datasetu* *oraz* *ClipSeg* *do*
*automatycznego* *maskowania* *obrazów.)*

**Przygotowanie** **danych** **do** **treningu**

Przygotowanie datasetu to najważniejszy etap trenowania modelu
(zwłaszcza LoRA). Tutaj podsumujemy najlepsze praktyki w kontekście
OneTrainer:

**Obrazy** **i** **podpisy** **(.txt)**

Zbiór treningowy powinien składać się z obrazów powiązanych z
tematem/obiektem, którego model ma się nauczyć. Do każdego obrazu
**rekomenduje** **się** **przygotować** **opis** **tekstowy** (tzw.
*caption*). Choć OneTrainer formalnie pozwala trenować *bez* *podpisów* (model
wtedy próbuje nauczyć się wzmocnienia cech czysto z obrazów, co bywa
nieprzewidywalne), **dobry** **opis** **do** **każdego** **obrazka**
**zdecydowanie** **poprawia** **efekt**. Kilka wskazówek:

 • **Format** **plików:** używaj powszechnych formatów graficznych
 (JPEG, PNG). Rozdzielczości – najlepiej powyżej 512px w obu wymiarach
 (im większe, tym lepiej, ale też dłużej się trenuje i więcej VRAM
 zużywa; typowo 512–1024px). OneTrainer i tak będzie skalował obrazy do
 wybranych bucketów rozdzielczości, np. 512x512, 640x360, 1024x576
 etc., więc nie musisz ich ręcznie skalować – ważne by były
 wystarczająco duże i ostre.

 • **Parowanie** **z** **.txt:** nazwy plików tekstowych powinny
 dokładnie odpowiadać nazwom obrazów. Jeśli masz photo123.jpg , stwórz
 photo123.txt z opisem. Gdy obraz jest .png , nazwa .txt również
 .png.txt lub .txt (zależy od OS, zwykle .png i .txt to różne
 rozszerzenia, więc

 14

 dokładnie photo.png i photo.png.txt nie będą uważane za parę – unikaj
 kropek w nazwie poza rozszerzeniem).

 • **Treść** **podpisów:** staraj się, by były **opisowe** **i**
 **zawierały** **istotne** **szczegóły**. Reguła: *zamieść* *w*
 *opisie* *wszystko,* *co* *może* *się* *zmieniać* *między* *obrazami,*
 *a* *pomiń* *to,* *co* *jest* *wspólne*. Np. trenując LoRA konkretnej
 postaci – w opisach zawrzyj jej ubiór, pozę, scenerię, ale **nie**
 **powtarzaj** jej imienia w każdym (to nauczy model, że imię=ta
 postać, co jest ok, ale jeśli LoRA ma być wywoływana unikalnym tokenem
 to lepiej dodać go raz w każdym podpisie). Unikaj też zbyt
 technicznych tagów, jeśli nie są potrzebne. **Jakość** **ponad**
 **ilość** – lepiej mniej tagów, ale trafnych. Według niektórych
 źródeł, 5-15 słów kluczowych to optimum dla jednego obrazka.

 • **Konsystencja** **nazw:** Jeżeli trenujesz koncept (np. nową osobę
 czy styl), często stosuje się *unikalny* *token* – np. słowo typu
 XYZperson w każdym podpisie, by model nauczył się kojarzyć to słowo z
 celem. W DreamBooth np. każdy podpis zawiera frazę „person XYZ” obok
 reszty opisu. W LoRA też można tak robić (zwłaszcza styl LoRA – często
 w embeddingach/LoRA stylu stosuje się unikalny token).

 • **Jakość** **i** **różnorodność** **obrazów:** Obrazy powinny
 pokazywać temat w różnych ujęciach, oświetleniu, tle itp. Unikaj
 duplikatów i prawie identycznych ujęć – nie wnoszą nowej informacji a
 mogą spowodować przeuczenie. Lepiej mieć 50 zróżnicowanych zdjęć niż
 200 bardzo podobnych.

 • **Ile** **obrazów?** To zależy od złożoności konceptu. LoRA potrafi
 nauczyć się prostej koncepcji (np. stylu rysunkowego) nawet z ~20–30
 obrazków. Dla osoby/postaci realistycznej lepiej mieć 50–100. Ogólnie
 100–200 to często górna granica sensowności (powyżej tego model 1.5
 raczej już zapamięta dość, a LoRA staje się cięższa). Dla SDXL można
 dać więcej, bo model jest większy – np. 200–500 obrazów różnych może
 być OK, ale to też wydłuża trening.

 • **Walidacja:** Przygotuj kilka obrazów (nawet 5–10) **spoza**
 **treningu** do walidacji, z analogicznymi podpisami. To powinny być
 obrazki reprezentatywne dla koncepcji, ale **nieużywane** **w**
 **treningu** – wtedy walidacja pokaże na nich loss. Jeśli loss
 walidacyjny zaczyna rosnąć, wiesz że model zaczyna przeuczać
 oryginalne dane kosztem generalizacji. Nie jest to obowiązkowe, ale warto, zwłaszcza przy większych
 projektach.

Powyższe można streścić: *dobre* *dane* *=* *dobry* *LoRA*. OneTrainer
udostępnia narzędzia, by Ci to ułatwić
(auto-caption,taggerwTools).Skorzystajznich,alezawsze**zweryfikujautomatyczniewygenerowane**
**opisy** – np. WD14 może dodać tagi, które nie pasują (usuń je), BLIP
może opisać scenę za bardzo ogólnie (doprecyzuj ręcznie). Zainwestowany
czas przed treningiem zaoszczędzi Ci frustracji po.

**Augmentacje** **obrazów**

Augmentacje to sposób na sztuczne powiększenie różnorodności datasetu.
OneTrainer, jak opisano w zakładce Concepts → Image Augmentation,
pozwala włączać różne losowe transformacje: odbicia, rotacje,
przycięcia, zmiany jasności, kontrastu itp. Zalecenia:

 • **Używaj** **augmentacji,** **gdy** **masz** **mało** **danych.**
 Jeśli masz <50 obrazów i widzisz, że np. wszystkie są w podobnym
 otoczeniu, augmentacje mogą pomóc modelowi nie przywiązywać się do
 specyfiki tła czy kolorystyki. Np. dodaj losową zmianę jasności ±10%,
 losowy flip.

 • **Nie** **przekombinuj.** Zbyt agresywne augmentacje (np. obrót o
 90°, duże zmiany hue) mogą utrudnić modelowi nauczenie się właściwego
 konceptu, bo wprowadzą sprzeczne informacje. Chcesz raczej drobnych
 losowych wahań – tak by model nauczył się, że to nieistotne dla
 koncepcji.

 • **Crop** **jitter** **jest** **przydatny** **zawsze,** gdy Twoje
 obrazy nie mają identycznych wymiarów. Model dostanie różne
 wykadrowania – to dobrze.

 • **Wyłącz** **augmentation,** **gdy** **nie** **potrzeba.** Jeśli
 Twój zbiór jest spory i różnorodny, augmentacje mogą nie być
 konieczne, a wydłużą trening (każdy obraz co epokę jest trochę inny –
 niby dobrze,ale może utrudniać konwergencję). Czasem lepiej trenować więcej epok
 na czystych danych niż mniej epok z augmentacjami – zależy od
 przypadku.

Przykład: trenujesz LoRA stylu malarskiego i masz tylko 10 obrazów –
warto włączyć flips, rotacje ±5°, drobny color jitter (±0.05 hue, ±0.1
saturacji). Dzięki temu model nie nadmiernie dopasuje się np. do
dominującego koloru oświetlenia na tych 10 obrazach.

Jeszcze uwaga: augmentacje w OneTrainer mogą być *deterministyczne* lub
*losowe* co epokę. Domyślnie są losowe (i tak jest ok). Gdy są losowe,
pamiętaj o interakcji z cache (Image/Text Variations ustaw jak w
Concepts wyżej).

**Proporcje** **obrazu** **i** **bucketing** **(Aspect** **Ratio**
**Buckets)**

Trening generatywny preferuje obrazy kwadratowe (512x512, 768x768,
itp.), bo takie były oryginalne dane SD1.x. Jednak świat nie jest
kwadratowy – Twoje dane mogą mieć różne proporcje. **Aspect** **Ratio**
**Bucketing** to rozwiązanie tego problemu, które OneTrainer wdraża
automatycznie (gdy włączone). Jak to działa i co warto wiedzieć:

 • Gdy AR Buckets są włączone, musisz ustawić **listę** **docelowych**
 **rozdzielczości** (bucket resolutions). W presetach OneTrainer ma to
 ustawione rozsądnie. Np. dla SD1.5 LoRA możesz zobaczyć listę:
 512x512, 512x640, 640x512, 512x768, 768x512 itd. – pokrywającą typowe
 proporcje (1:1, 4:3, 3:4, 2:3, 3:2, 16:9...).

 • OneTrainer przy wczytywaniu datasetu zmierzy wymiary każdego obrazu
 i przypisze obraz do najbliższego “wiaderka” (bucket) spośród
 zadanych, tak aby liczba pikseli mniej więcej się zgadzała
 
 . Następnie podczas treningu obrazy będą skalowane do wymiarów bucketu
 (z zachowaniem aspektu, reszta jest docinana z włączonym jitter jeśli
 zaznaczyłeś).

 • Dzięki temu, np. panoramy 16:9 będą trenowane na bucketach szerokich
 (np. 1024x576), a portrety 3:4 na bucketach pionowych (np. 512x682) –
 model uczy się obu proporcji. To ogromna przewaga nad starszym
 podejściem, gdzie musiałbyś albo przyciąć wszystko do kwadratu, albo
 dodać czarne paski.

W praktyce: - **Dodawaj** **wszystkie** **sensowne** **buckety** –
OneTrainer chyba generuje listę bucketów automatycznie na podstawie
datasetu (można to modyfikować). Upewnij się, że skrajne proporcje są
objęte, bo jak nie znajdzie bliskiego bucketu, to i tak przypisze do
najbliższego (co może oznaczać większe przycięcie niż byś chciał). -
**Docelowa** **rozdzielczość:** Wybierz maksymalny wymiar bucketów
adekwatny do modelu i VRAM. Dla SD1.5 typowo używa się 512 lub 640 px
jako podstawy. Dla SDXL – 1024 px. Model SDXL jest trenowany do
generowania 1024x1024, więc LoRA na SDXL warto trenować w rozdziałce
zbliżonej (np. bucketi 1024 i mniejsze). To wymaga ~12GB VRAM na
batch=1, rank=8, więc posiadacze 8GB mogą zejść do 768px – ale to
kompromis, LoRA trenowana na niższym rozmiarze może działać trochę
gorzej na wyższych. - **Przy** **bucketingu** **loss** **liczone**
**jest** **per** **bucket.** Możesz w TensorBoard zobaczyć stratę w
zależności od bucketu (to dla dociekliwych). Jeśli jakiś bucket ma
wyraźnie większy loss, może to znaczyć, że model ma trudność z tą
proporcją. Często jednak to po prostu powód: mniej danych w tym
formacie. - **Nie** **musisz** **nic** **specjalnego** **robić**
**oprócz** **włączenia** **AR** **Buckets** – OneTrainer zajmie się
resztą.

Podsumowując: bucketing to przyjaciel, zostaw go włączonym chyba że masz
**wszystkie** obrazy o identycznej wielkości. Dzięki bucketom model LoRA
będzie funkcjonował dobrze dla różnych rozdzielczości generacji (np.
zarówno portret 9:16, jak i pejzaż 16:9), co jest bardzo pożądane.

*(Dodatkowe* *szczegóły* *można* *znaleźć* *na* *wiki* *OneTrainer:*
*stronę* *Aspect* *Ratio* *Bucketing*


**Trenowanie** **modelu** **LoRA** **–** **przykłady**

Mając przygotowane dane i skonfigurowany OneTrainer, przechodzimy do
właściwego treningu. Poniżej przedstawiamy dwa scenariusze: trenowanie
LoRA na bazie modelu **Stable** **Diffusion** **XL** (SDXL) oraz
trenowanie LoRA na bazie modelu **Flux** (specyficzna architektura typu
DiT).

**LoRA** **na** **bazie** **SDXL**

Załóżmy, że chcesz stworzyć LoRA, która pozwoli generować obrazy
stylizowane na określone dzieła sztuki, albo przedstawiające konkretną
osobę – a bazą ma być nowoczesny model **SDXL** **1.0**. Wykonaj
następujące kroki:

 1. **Przygotuj** **dane** **i** **konfigurację:** Zgromadź dataset
 (obrazy + podpisy) zgodnie z poradami powyżej. Następnie w GUI
 OneTrainer **wybierz** **preset** **SDXL** **LoRA**. W lewym górnym
 rogu, na liście ‘configs’, powinien być dostępny np. szablon „SDXL
 LoRA” – wybierz go. Spowoduje to automatyczne ustawienie wielu parametrów pod SDXL (np.
 odpowiednia architektura modelu, parametry optymalizatora i buckety
 1024px). Jeśli nie ma takiego presetu domyślnie, możesz skorzystać z
 publicznie dostępnych – niektórzy udostępniają configi OneTrainer (np.
 Furkan G. na swoim GitHub/Gumroad).

 Upewnij się, że w zakładce **General** ustawiłeś *Workspace*
 *Directory* (np. workspace/ SDXL_loRA_01 ) – tam trafią wyniki. Możesz
 zwiększyć *dataloader* *threads* jeśli masz mocny CPU (SDXL dość wolno
 ładuje dane, bo obrazki duże, wątkowanie pomaga).

 2. **Model** **bazowy** **SDXL:** W zakładce **Model** wskaż *Base*
 *Model* na **SDXL** **base**. Masz kilka opcji: 3. Podaj link do
 modelu diffusers SDXL na HuggingFace (wymaga zalogowania lub tokenu bo
 to gated). Np. stabilityai/stable-diffusion-xl-base-1.0 (SDXL Base) lub
 wersję *Refiner* jeśli wolisz. *Uwaga:* SDXL Base to model 2.6B,
 potrzebuje sporo VRAM.

 4. Jeśli pobrałeś wcześniej plik SDXL (tzw. All-in-One safetensors z
 civitai czy huggingface) – wskaż do niego ścieżkę pliku .safetensors. AIO oznacza, że zawiera UNet + VAE + encodery w jednym, OneTrainer
 to obsłuży.

 5. Pamiętaj o **HuggingFace** **Token** – SDXL jest gated, więc wpisz
 swój token w polu HF Token w zakładce Model, inaczej OneTrainer nie pobierze modelu (chyba że wskazujesz lokalny
 plik).

 6. **Zakładka** **LoRA:** Sprawdź ustawienia:

 7. **Rank:** SDXL LoRA zwykle rank 8 lub 16. Na start proponuję rank
 **8** (mniej VRAM, mniejsze ryzyko przeuczenia, a wciąż daje
 zauważalny efekt).

 8. **Alpha:** zostaw 1.0 (OneTrainer i tak mnoży 1/8 = 0.125 w LR).

 9. **LoRA** **Type:** LoRA (klasyczna). Jeśli czujesz się na siłach,
 możesz spróbować LoHa, ale to bym zostawił na później.

 10. **Layers:** wybierz *Attn+MLP* (OneTrainer może mieć to domyślnie
 dla SDXL LoRA). To pozwoli LoRA modyfikować również warstwy MLP, co
 bywa potrzebne przy stylach/kolorach w SDXL.

 11. **Dropout:** zacznij od 0 (brak dropout). Jeśli zobaczysz w
 trakcie, że LoRA mocno przeucza, przerwiesz i ustawisz np. 0.1.

 12. **LoRA** **base** **model** **(resume):** puste, bo trenujemy od
 zera LoRA. 13. **Reszta** – weight dtype zostaw float32.

 14. **Ustaw** **hyperparametry** **Training:** Preset SDXL LoRA
 zapewne ma optymalizator AdamW i jakiś LR rzędu 1e-4 lub adaptacyjny
 optymalizator.

 15. **Batch** **size:** jeśli masz GPU 24GB, możesz dać batch 2 lub 4
 dla 1024px. Na 12GB pewnie tylko 1. Dostosuj tak, by VRAM usage ~90%.

 16. **Epochs:** Zależy od liczby obrazów. Załóżmy masz 100 obrazów.
 LoRA często potrzebuje 10-20 epok by dobrze nauczyć koncept (SDXL
 model jest „świeży”, może wymagać mniej bo jest potężniejszy). Ustaw
 np. 10 epok, po 10 epokach zobaczysz czy już efekt OK.

 17. **Save** **every** **epoch** – tak będzie bezpiecznie.

 18. **Learning** **rate:** jeśli używasz adaptacyjnego optymalizatora
 (dAdaptation, Prodigy), nie musisz dużo zmieniać (one się dostosują).
 Jeśli AdamW – ustaw np. 1e-4 start i scheduler Cosine decaying do
 1e-6.

 19. **Text** **Encoder** **training:** w SDXL raczej nie ruszaj
 encodera, chyba że Twój koncept to coś mocno tekstowego (jak nowe
 słowo, ale to raczej embedding). Możesz w OneTrainer zostawić Train
 Text Encoder = False (domyślnie LoRA preset chyba i tak nie trenuje
 go).

 20. **Validation:** jeżeli przygotowałeś walidacyjny concept, włącz
 Validation (General tab) i ustaw np. Validate after 1 epoch, by co
 epokę liczył loss walidacyjny.

 21. **Sampling:** ustaw sobie kilka promptów do testu z tokenem
 Twojej LoRA, co 1 epoch generuj np. 2 obrazy (to zajmuje trochę czasu,
 ale warto).

 22. **Rozpocznij** **trening:** Kliknij **Start** **Training** (duży
 przycisk w UI). OneTrainer zacznie proces – w konsoli (dolny panel)
 zobaczysz logi. Powinno się pojawić info o wczytywaniu modelu bazowego
 (może chwilę zająć, SDXL jest duży). Następnie tworzenie cache (jeśli
 włączony latent caching) – pierwszy epoch może być wolniejszy przez
 to. Potem iteracje treningowe.

 23. **Monitoruj:** Obserwuj *loss* w konsoli lub w TensorBoard
 (kliknij *TensorBoard*). Loss powinien spadać stopniowo. Oglądaj też
 generowane *sample* *images* – to najlepszy wskaźnik. Początkowo
 (epoch 1-2) będą losowe bzdury, ale z epoki na epokę powinny coraz
 bardziej przypominać oczekiwany rezultat. Gdy uznasz, że wygląd jest
 satysfakcjonujący, **możesz** **przerwać** **trening** **wcześniej** –
 nie trzeba wyciskać wszystkich epok, jeśli np. po 8 epokach model już
 ładnie generuje to, co chcesz. Z drugiej strony, jeśli po zadanej
 liczbie epok wynik jest zbyt słaby lub wciąż się poprawia – możesz
 dołożyć kolejnych epok (wystarczy zwiększyć wartość i trening poleci
 dalej, lub przerwij i wznów z Continue backup).

 24. **Zakończenie:** Po zakończeniu OneTrainer zapisze finalny plik
 LoRA (zgodnie z *Model* *Output*). Znajdziesz go w określonym miejscu
 (np. models/mojaLora.safetensors ). Teraz najważniejsze –
 **przetestuj** **LoRę**! Wgraj ją do ulubionego narzędzia
 generatywnego (np. Automatic1111, ComfyUI). W promptach użyj
 tokenów/deskryptorów, jakich uczyłeś i sprawdź czy efekt jest zgodny z
 oczekiwaniami.

*(Przykład:* *Wytrenowaliśmy* *LoRę* *stylu* *malarskiego* *VanGogh.*
*Ładujemy* *ją* *w* *Auto1111,* *prompt:* *„portrait* *of* *a* *man,*
*in* *<wynikowy* *styl* *style”,* *weight* *0.8* *–* *i*
*otrzymujemy* *obrazy* *przypominające* *Van* *Gogha.)*

OneTrainer ułatwia cały powyższy proces – wystarczy poprawnie ustawić
opcje. W razie wątpliwości zawsze zerknij do dokumentacji wiki lub do
społeczności na Discordzie OneTrainer (link w repo).

**Tip:** Trenowanie LoRA na SDXL wymaga sporo VRAM. Jeśli masz tylko 8
GB, rozważ użycie *ofloading* (Temp device = CPU) i ewentualnie
obniżenie precision do bfloat16 lub int8 – to pozwoli zmieścić SDXL
trening, ale będzie wolniej. Alternatywnie możesz trenować SDXL LoRA w
chmurze (RunPod, Colab etc.) – OneTrainer ma nawet poradniki jak to
skonfigurować.

**LoRA** **na** **bazie** **modelu** **Flux**

Model **Flux** (dokładnie Flux.1) to dość eksperymentalna architektura
od Black Forest Labs – *Flow* *Matching* *Diffusion*. Ma duży potencjał,
ale jest znacznie większy i wolniejszy od SD1.x/SDXL. Załóżmy, że chcesz wytrenować LoRA na modelu Flux, np. żeby generować
rzadki typ obrazów (Flux
potrafigenerowaćemotikony,bobyłtrenowanym.in.naprostegrafiki).Wyzwanie:oficjalnieOneTrainer
wspiera Flux, ale dokumentacja jest szczątkowa. Oto kroki:

 1. **Przygotuj** **model** **Flux** **localnie:** Model Flux.1-dev
 jest udostępniony na HuggingFace (black-forest-labs/FLUX.1-dev).
 Jednak aby użyć go w OneTrainer, najlepiej mieć go w formacie
 diffusers. Sposób:

 2. Zaloguj się na HuggingFace i uzyskaj **User** **Access** **Token**
 (w Profil - Settings - Access Tokens). Wklej go w OneTrainer
 zakładce Model (HF Token).

 3. W OneTrainer, zakładka Model: jako Base Model wpisz **HF**
 **link** **do** **Flux**: black-forest-labs/ FLUX.1-dev . Jeśli token
 jest poprawny i masz dostęp (Flux.1-dev raczej wymaga akceptacji
 warunków?), OneTrainer powinien spróbować pobrać model. **UWAGA:** Ten
 model to ~19GB danych! Lepiej pobrać go ręcznie:

 ◦ Wejdź na huggingface link i pobierz *wszystkie* *pliki* *i*
 *foldery*, szczególnie foldery: scheduler , text_encoder ,
 text_encoder_2 , tokenizer , tokenizer_2 , transformer , vae oraz plik
 model_index.json. (Nie potrzebujesz pliku flux1-dev.safetensors z głównego katalogu –
 on jest łączony z powyższymi i tak).

 ◦ Umieść to w lokalnym folderze, np. models/FLUX.1-dev/ tak, by
 wewnątrz były te podfoldery i model_index.json.

 ◦ W OneTrainer Base Model wskaż ten folder.

 4. Alternatywnie, BFL (autorzy Flux) udostępniali też All-in-One
 safetensors (helheimFlux_v10FP16AIO.safetensors itp.). OneTrainer powinien je obsłużyć (przy Base Model wskaż plik
 .safetensors). Upewnij się tylko, że to wersja *AIO* *z* *text*
 *encoderami*. Format NF4 AIO nie jest wspierany, Turbo wersji flux też nie użyjesz (inny format), FLEX (inna arch.
 pokrewna Flux) również nieobsługiwane. Zalecany jest oryginalny Flux.1 dev FP16 AIO lub diffusers format.

 5. **Konfiguracja** **OneTrainer** **dla** **Flux:** Wybierz preset
 **FluxDev** **+** **LoRA**. OneTrainer od wersji ~0.4 miał takie
 presety (na górnym pasku, kombinuje „FluxDev” i tryb LoRA). Po wybraniu upewnij się:

 6. W General → **Train** **device** = cuda, **Temp** **device** =
 cpu. Flux jest ogromny, więc domyślnie OneTrainer włączy ofload na CPU
 (inaczej raczej VRAM nie starczy). Możesz spróbować upchnąć wszystko w
 VRAM, ale autorzy sugerują, że 12GB VRAM jest minimalne, 8GB to już z
 trudem nawet z ofloadem.

 7. W Model → Base Model powinien być już wskazany (jeśli preset to
 zrobił). Jeśli nie, podaj jak wyżej (folder).

 8. **HF** **Token** wpisany jeśli model wymaga.

 9. *Model* *Output* daj nazwę pliku np. flux_lora.safetensors .

 10. Data → AR Buckets: Flux natywnie generuje 512x512 (o ile wiem),
 ale może wspiera wyższe? Bezpiecznie daj bucket 512 i ewentualnie parę
 innych (chyba w presecie jest).

 11. Concepts → dodaj swój dataset (podobnie jak dla SDXL, podpisy
 itd.). Flux to inny model tekstowy (T5 XXL encoder), ale podpisy w
 formie tagów czy zdań – jedno i drugie działa. Z doświadczeń: TEnc2
 (OpenCLIP G) nie istnieje w Flux; flux ma T5 XXL. Więc raczej nastaw
 się, że *embeddingi* *tekstowe* *oryginalnego* *SDXL* *tu* *nie*
 *mają* *analogu* – generowanie tekstu jest inne. Mimo to, opisy
 tekstowe w dataset do LoRA oczywiście daj jak zwykle.

 12. LoRA tab:

 ◦ **Rank:** Flux LoRA zalecany jest rank 16 (mimo że model duży).
 Mniejsze rank pewnie też zadziała, ale społeczność najczęściej rank 16
 używała.

 ◦ **Precision:** tu ważne – **Flux** **wymaga** **wysokiej**
 **precyzji.** BFL zaleca FP8 co najmniej
 
 . OneTrainer obsługuje FP8 i NF4 – możesz w Model tab  Data Types
 wybrać weight dtype = FP8 (lub NF4). NF4 (4-bit) umożliwi trenowanie
 na niższym VRAM, ale daje artefakty (grid pattern). FP8 jest lepsze jakościowo, ale 8-bit wagi to wciąż duży model.
 Jeśli masz 8GB VRAM, pewnie musisz NF4 + ofload sporo.

 ◦ **Optimizer:** preset pewnie ustawi AdaFactor lub AdamW8bit –
 Adafactor jest wskazany dla dużych modeli (bo oszczędza pamięć).

 ◦ **DoRA:** Nie wiemy, pewnie można spróbować, ale to ryzykowne – flux
 i tak trudny. Zostaw off.

 13. Training tab:

 ◦ **Batch** **size:** flux jest ogromny i wolny. Batch 1 to najpewniej
 max, nawet na 24GB.

 ◦ **Steps/Epochs:** flux uczy się powoli. Niestety brak tu szerokich
 doświadczeń – zacznij od np. 1000 steps i zobacz. W logach flux bywa
 ~1.5x wolniejszy od SDXL. Może 1000 steps starczy, może potrzeba 3000
 – zależy od danych.

 ◦ **LR:** Najczęściej używano tu Adafactor z relative decay – to
 optymalizator bez sztywnego LR. Jeśli używasz Adam, spróbuj bardzo
 mały LR (1e-5?) bo flux może eksplodować zbyt dużym LR.

 ◦ **Mask/No** **mask:** raczej normalne treningi, maski działają tak
 samo jak w SDXL.

 ◦ **Validation**: flux raczej nie, bo i tak wolno trenuje – szkoda
 czasu. Skup się na finalnym wyniku.

 14. **Sample** **images:** Uwaga – flux generuje inne latenty? Nie,
 ma VAE więc powinien generować. Ale generowanie z flux w trakcie
 treningu może być bardzo wolne (flux transformera jest heavy). Możesz
 to wyłączyć by nie przedłużać. Ewentualnie wygeneruj 1 obraz co 1
 epoch.

 15. **Trening:** Start Training. Bądź cierpliwy – ładowanie flux to
 sporo (transformer ~5GB model). Możliwe, że OneTrainer trochę dłużej
 będzie inicjalizował. Jeśli wybrałeś precision NF4 lub FP8, i masz
 cuda+cpu ofload, monitoruj RAM (może zużywać dziesiątki GB).

 16. Loss fluxa może być wyższy niż w SD (inny zakres). Nie zrażaj
 się, ważne by spadał.

 17. OneTrainer dev-corner wspomina, że LoRA to jedyny rekomendowany
 sposób trenowania Flux (pełny finetune jest niepraktyczny, bo wymaga
 też destylacji modelu).

 18. Ponadto flux LoRA **będzie** **działać** **tylko** **w**
 **ComfyUI** **lub** **w** **OneTrainer** **obecnie**, bo format LoRA
 dla flux jest inny niż standard SD LoRA (inne klucze tensorów).
 ComfyUI obsługuje LoRA fluxowe (ale np. Auto1111 pewnie nie, dopóki
 nie zaimplementują). Więc po treningu testuj wynik w ComfyUI z loaderem LoRA – developer
 wspominał, że standardowy loader LoRA w ComfyUI obsłuży flux LoRA.

 19. **Zakończenie:** Po treningu otrzymasz plik .safetensors.
 Przetestuj go. Flux generuje nieco inne style niż SD – upewnij się, że
 to co chciałeś wyszło. Być może okaże się, że trzeba było dać rank 32
 – flux ma *bardzo* *robust* *architecture*, LoRA 512 czy 768 może
 generować 1024 z minimalną stratą jakości. To znaczy flux LoRA jest potencjalnie dość generalna.

 20. Jeżeli masz problem, że Twój flux LoRA nie działa w innym
 oprogramowaniu – to normalne, dopóki tamto nie wspiera. Możesz użyć
 narzędzia convert (OneTrainer convert_model.py) by spróbować
 przekonwertować LoRA flux na format kompatybilny, ale raczej to
 kwestia czasu aż integracje powstaną.

Trening na flux to **zaawansowany** **temat**. Zaleca się do niego
podejść dopiero mając doświadczenie z normalnymi LoRA. Niemniej,
OneTrainer czyni to możliwym. Wsparcie community jest ograniczone (Flux
to nowość z 2025), więc posiłkuj się wątkiem na Reddit i wiki flux.

*(Pro-tip:* *do* *eksperymentów* *z* *flux* *można* *użyć*
*akceleratora* *z* *dużą* *VRAM* *na* *chwilę* *w* *chmurze* *–*
*przyspieszy* *to* *próby.* *Potem* *LoRę* *możesz* *używać* *lokalnie*
*do* *generacji,* *bo* *generować* *fluxem* *da* *się* *na* *8GB* *przy*
*512x512* *jednym* *kroku.)*

**Zastosowanie** **wytrenowanych** **modeli** **w** **innych**
**narzędziach**

Wytrenowane za pomocą OneTrainer modele można z powodzeniem
wykorzystywać w popularnych narzędziach do generowania obrazów AI,
takich jak **Automatic1111** **(Stable** **Diffusion** **WebUI)**,
**InvokeAI**, **Fooocus**, **ComfyUI** i inne. OneTrainer zapisuje
modele w standardowych formatach ( .safetensors lub .ckpt ), dzięki
czemu integracja jest prosta:

 • **LoRA** **w** **WebUI** **(Auto1111):** Po ukończeniu treningu LoRA
 umieść plik .safetensors w folderze stable-diffusion-webui/models/Lora
 . W interfejsie WebUI możesz potem w promptach używać swej LoRA za
 pomocą słowa kluczowego <lora:nazwa:wagę . Np. jeśli LoRA nazywa
 się mystyle.safetensors , użyj '<lora:mystyle:0.8' w prompt.

 Auto1111 obsługuje zarówno LoRA typowe, LoHa, jak i embedowane textual
 inversions (jeśli Bundle Embeddings było włączone) – zostaną one
 odczytane. *Uwaga:* w przypadku LoRA typu DoRA może być potrzebna najnowsza
 wersja WebUI lub skryptu dodatkowego, bo DoRA to nowość (ale
 generalnie powinna działać, to tylko inna zawartość wag).

 • **InvokeAI/Fooocus:** Te narzędzia również wspierają LoRA (format
 Diffusers). Możesz zaimportować LoRA plik do ich interfejsu. InvokeAI
 ma możliwość wczytania LoRA i stosowania jej podczas generacji.
 Fooocus – jeśli obsługuje LoRA, to zapewne także poprzez wczytanie
 pliku (ew. trzeba spakować do format .pt Diffusers?). Sprawdź
 dokumentację konkretnego narzędzia, ale z reguły .safetensors LoRA z
 OneTrainer będzie kompatybilny.

 • **ComfyUI:** ComfyUI obsługuje LoRA poprzez nod *Load* *LoRA*. Plik
 z OneTrainer wczytasz tam i podłączysz do modelu. Co więcej, ComfyUI
 już obsługuje *różne* *formaty* *LoRA* *OneTrainer* (w tym LoHa i
 DoRA) oraz nawet LoRA do Flux. Tak więc Comfy to świetne środowisko do testowania zaawansowanych
 LoRA.

 • **Wytrenowane** **pełne** **modele** **(fine-tune):** OneTrainer
 potrafi trenować pełne checkpointy (choć tu nie skupialiśmy się na
 tym). Jeśli użyjesz trybu *Full* *finetune* i zapiszesz model jako
 safetensors/ ckpt, to taki plik możesz bezpośrednio umieścić w
 models/Stable-diffusion w Auto1111 lub odpowiadającym folderze w
 InvokeAI i używać jak normalny model. Np. możesz wytrenować
 *DreamBooth* model w OneTrainer (to właściwie full fine-tune z prior,
 podobnie jak kohya) i użyć go do generacji portretów w WebUI.

 • **Kontenery** **(OMI)**: W przyszłości może pojawić się format OMI
 (One Model Integration) do przenoszenia embeddingów z Flux do innych –
 na razie to ciekawostka.

Krótko mówiąc, **OneTrainer** **jest** **kompatybilny** **z**
**ekosystemem** **Stable** **Diffusion**. Jego zaletą jest unifikacja
formatu – trenujesz w nim, a potem korzystasz tam, gdzie wygodnie.
Twórcy LoRA często używają kohya SS; OneTrainer daje alternatywę, a
wyniki (pliki) są **wymienne** między tymi narzędziami. Możesz też
porównać jakość: np. LoRA stworzona w OneTrainer vs LoRA stworzona w
kohya – wczytaj obie do WebUI i porównaj na tych samych promptach. Wielu
użytkowników zauważa, że OneTrainer potrafi dać równie dobre, a czasem
lepsze rezultaty (lepsza obsługa maskowania, dropout, optymalizatorów).

Na koniec: nie zapomnij się **podzielić** **swoim** **modelem** jeśli
jest ciekawy! Format safetensors jest bezpieczny – możesz go opublikować
na CivitAI czy HuggingFace, opisując jak trenowałeś (tu możesz
wspomnieć, że użyłeś OneTrainer – to pokaże, że masz praktyczne
doświadczenie z nowoczesnymi narzędziami ML).

**Tryb** **CLI** **–** **skrypty** **i** **ich** **zastosowania**

OneTrainer w trybie konsolowym oferuje bogaty zestaw skryptów Python,
które pozwalają automatyzować i rozszerzać działania, które normalnie
wykonujesz przez GUI. Oto najważniejsze skrypty dostępne w folderze scripts/ oraz przykłady
ich użycia:

 • **train.py** **:** Główny skrypt treningowy. Pozwala uruchomić
 trening bez GUI – wszelkie parametry podajesz jako argumenty. Jest
 szczególnie przydatny na serwerach (gdzie nie masz ekranu) lub do
 pisania własnych procedur. Przykład użycia:

 python scripts/train.py --config configs/SDXL_LoRA.json

 (zakładając, że zapisaliśmy konfigurację z GUI do pliku JSON) lub:

 python scripts/train.py --base_model
 "stabilityai/stable-diffusion-2-1" --train_data_dir "data/images"
 --output_dir "models/output" --lora_rank 8 [inne opcje...]

 Parametrów jest dużo – aby je poznać, uruchom python scripts/train.py
 -h
 
 . Warto podkreślić, że **GUI** **OneTrainer** **jest** **nakładką** na
 ten skrypt – konfigurując w UI i klikając Start, tak naprawdę
 uruchamiany jest train.py z odpowiednimi argumentami. Dlatego
 przejście na CLI jest naturalne, gdy np. chcesz wykorzystać klaster
 HPC do treningu.

 • **train_ui.py** **:** Alternatywny skrypt uruchamiający mini-UI w
 przeglądarce do monitorowania treningu w trybie CLI. Można odpalić
 OneTrainer na serwerze bez X11, a w przeglądarce na PC otworzyć port i
 mieć podgląd.

 • **create_train_files.py** **:** Pomocnik do generowania plików
 konfiguracyjnych i struktur folderów gdy chcesz trenować z CLI bez
 interakcji. Może utworzyć szablony .json z konfiguracją trenowania,
 które potem edytujesz.

 • **generate_captions.py** **:** Skrypt do automatycznego tagowania
 obrazów (captioning)
 
 . To odpowiednik funkcji w Tools GUI, ale z poziomu konsoli. Obsługuje
 BLIP, BLIP2, WD14 – parametry pozwalają wybrać model, ścieżkę do
 obrazów, prefixy itd. Przykład:

 python scripts/generate_captions.py --image_dir "data/mydataset"
 --caption_model "blip2" --device cuda

 Spowoduje wygenerowanie plików .txt dla obrazów w katalogu, używając
 BLIP2 (który zostanie pobrany przy pierwszym użyciu). Możesz też
 skorzystać z WD14 tagger:

 python scripts/generate_captions.py --image_dir "data/mydataset"
 --caption_model "wd14-convnextv2" --append_tags --tag_threshold 0.35

 To wygeneruje tagi Danbooru dla obrazków, dołączając je (append) do
 istniejących podpisów, z progiem pewności 0.35. Skrypt jest bardzo
 elastyczny – użyteczny gdy masz wiele folderów do otagowania. (Zwróć
 uwagę na ewentualną konieczność pobrania modelu WD14 – log powie co
 robić).

 • **generate_masks.py** **:** Podobnie jak wyżej, ale do
 automatycznego tworzenia masek
 
 . Pozwala wybrać metodę (clipseg, rembg) i przetworzyć cały folder
 obrazów generując maski. Np.:

 python scripts/generate_masks.py --image_dir "data/mydataset" --method
 "rembg"

 lub z użyciem ClipSeg z promptem:

 python scripts/generate_masks.py --image_dir "data/mydataset" --method
 "clipseg" --prompt "object in center"

 Skrypt ten oszczędza czas, gdy chcesz przygotować maski do masked
 training hurtowo. • **sample.py** **:** Narzędzie do generowania
 obrazów (inference) z dowolnego modelu, bez

 potrzeby użycia osobnego UI
 
 . Możesz np. wygenerować serię obrazów z modelem (będącym wynikiem
 treningu) w ramach ewaluacji. Przykład:


 python scripts/sample.py --model_path "models/MyLora.safetensors"
 --base_model "stabilityai/stable-diffusion-1-5" --prompt "a cat"
 --negative_prompt "blurry" --output_dir "outputs/samples"

 Powyższe załaduje model podstawowy 1.5, nałoży LoRę MyLora i
 wygeneruje obraz z zadanym promptem. Możesz oczywiście generować wiele
 obrazów, zmieniać seed, itp. Dobre do testowania LoRA w środowisku
 headless (np. automatyczny grid z różnymi wagami LoRA?).

 • **Inne** **skrypty:**

 • calculate_loss.py – policzy indywidualny loss dla każdego obrazu w
 dataset (przydatne do diagnozy, które obrazy są „trudne” do nauczenia).

 • Skrypty konwersji: convert_model.py , convert_model_ui.py –
 konwersja modelu między formatami (ckpt <- diffusers).

 • Debug: export_debug.py – generuje plik debug (log + env) do
 zgłaszania problemów.

 • Uruchamianie na Mac MPS czy AMD: dedykowane flagi lub skróty w
 launch scripts, jak opisano w dokumentacji launch (nie wgłębiamy się
 tutaj).

Generalnie, **wszystko** **co** **zrobisz** **w** **GUI,** **możesz**
**zrobić** **w** **CLI** i vice versa. GUI jest wygodne do interaktywnej
konfiguracji, CLI do automatyzacji i integracji z pipeline DevOps (np.
możesz mieć skrypt bash, który przygotowuje dane, odpala
generate_captions.py , potem train.py , a na końcu
publikujemodel–pełnaautomatyzacjatreningu).Dlarekruteramożetobyćotyleistotne,żeznajomość
trybu CLI świadczy o zrozumieniu, co dzieje się „pod maską” – np. wiesz
jak wywołać trening, jak parametry przełożyć na argumenty, itd.

*(Przykład* *użycia:* *Napisałeś* *własny* *kod* *generujący* *wariacje*
*datasetu* *i* *chcesz* *trenować* *wiele* *LoRA* *z* *różnymi*
*parametrami* *–* *możesz* *w* *Pythonie* *wywoływać* *train.py* *z*
*subprocess* *różnymi* *argumentami,* *logując* *wyniki.* *OneTrainer*
*CLI* *czyni* *to* *możliwym* *bez* *ręcznego* *klikania.)*

**Zaawansowane** **opcje** **OneTrainer**

W poprzednich sekcjach przewinęło się sporo zaawansowanych tematów.
Tutaj zbierzemy je w jednym miejscu i pokrótce omówimy, by podkreślić
Twoją znajomość tych aspektów:

**Ofloading** **do** **RAM** **(przenoszenie** **obciążenia** **na**
**pamięć** **RAM)**

OneTrainer potrafi automatycznie **ofloadować** część danych z GPU do
CPU, co jest zbawienne, gdy trenujemy bardzo duży model lub mamy
ograniczony VRAM. Mechanizm ten jest kontrolowany przez ustawienia
**Train** **device** i **Temp** **device** w zakładce General:

 • Jeśli **Temp** **device** **=** **CPU**, program będzie przenosił na
 RAM te części modelu, które nie są w danym momencie trenowane (np. w
 modelu SD część warstw U-Net może być przerzucana podczas trenowania
 encodera tekstowego i odwrotnie). Używa do tego biblioteki Accelerate
 od HuggingFace.

 • Ofloading wydatnie zmniejsza wymagania VRAM kosztem zwiększenia
 zużycia RAM i obciążenia magistrali CPU-GPU. Jak wspomniano, **64**
 **GB** **RAM** **jest** **zalecane** przy takich operacjach
 . Faktycznie, np. przy trenowaniu LoRA na SDXL 1.0 z 8GB VRAM, można
 zaobserwować zajętość RAM rzędu 40-50 GB, gdy wiele danych jest stale
 swapowanych.

 • Gdy **Temp** **device** **=** **cuda** (czyli to samo co train
 device), ofloading jest wyłączony – wszystko trzymane jest w VRAM. W
 praktyce przy małych modelach (SD1.5 z LoRA) tak jest optymalnie –
 overhead ofloadingu nie jest potrzebny.

 • OneTrainer stara się być sprytny: np. można ofloadować **EMA**
 **weights** na CPU, by nie zajmowały VRAM. Model może ofloadować tekstowe encodery gdy trenujemy U-Net i vice
 versa.

 • Dla bardzo dużych modeli (Flux, SDXL) ofloading to często jedyna
 opcja na średniej karcie. Lepiej, że program wolniej działa, niż jakby
 miał się nie uruchomić z braku pamięci.

Z punktu widzenia praktyki, **umiejętność** **ustawienia**
**ofloadingu** świadczy o zrozumieniu ograniczeń sprzętu. Np. w naszym
przewodniku przy Flux wskazaliśmy żeby użyć cuda+cpu, bo flux jest
ogromny – to pokazuje, że potrafisz dobrać strategie trenowania do
dostępnych zasobów.

Jeśli chodzi o implementację: Ofloading w OT oparty jest pewnie o
accelerate (z config Zero3 Ofload). Pamiętaj, że wymaga to pewnej
„stabilności” – np. w trakcie ofloadingu może dojść do *garbage*
*collection* i skokowych użyć pamięci (stąd 32GB to minimum, bo GC może
nie nadążyć i OutOfMemory).

Podsumowując: Ofloading to potężna funkcja, ale trzeba mieć na względzie
posiadany RAM. Przy 16GB RAM bym nie ryzykował; przy 32GB może zadziała,
lecz może swapować na dysk, co zabije wydajność.

**Precyzja** **obliczeń** **(precision)** **i** **typy** **danych**

Nowoczesne treningi często korzystają z obniżonej precyzji (mixed
precision) oraz kwantyzacji w celu zmniejszenia zużycia pamięci i
przyspieszenia obliczeń. OneTrainer daje sporo kontroli w tym zakresie w
zakładce Model (sekcja **Data** **Types**) oraz w opcjach LoRA.

 • **FP32** **vs** **FP16** **vs** **BF16:** Domyślnie modele mogą być
 trenowane w trybie mieszanej precyzji FP16 (float16) – niemal
 wszystkie obecne narzędzia to robią, bo FP32 jest zazwyczaj zbędne do
 stabilnej nauki i drogie pamięciowo. BF16 (bfloat16) to alternatywa do
 FP16, z pewnymi zaletami (szerszy zakres dynamiczny bez INF/NAN) –
 często preferowany na nowych GPU. OneTrainer potrafi użyć BF16 jeśli
 jest wspierane. W praktyce, na kartach Nvidia Ampere+ BF16 jest
 świetną opcją i w OT chyba tak jest ustawione np. w presetach SDXL.
 (Trening SDXL w BF16 jest bezproblemowy i bezpieczniejszy niż FP16).

 • **8-bit** **i** **4-bit** **(quantization):** OneTrainer integruje
 zapewne biblioteki bitsandbytes, co pozwala użyć 8-bitowych
 optymalizatorów (8-bit Adam) i przechowywania wag w 8-bitach. Ponadto
 wspiera formy kwantyzacji LoRA:

 • LoRA weight data type – wspomniana opcja, gdzie możesz wybrać np.
 int4 (NF4) dla wag LoRA
 
 . NF4 (Normalized Float 4) to 4-bitowy format dynamiczny, który
 zaskakująco dobrze sprawdza się do przechowywania modeli z minimalną
 utratą jakości. Użycie NF4 przy LoRA Flux było wręcz zalecane by
 zmieścić model.

 • Również model bazowy może być wczytany w formie 8-bit lub 4-bit,
 choć w Diffusers to nie jest natywne – OneTrainer mógłby użyć
 auto-kwantyzacji, ale najczęściej zostawia to optymalizatorowi.

 • **FP8:** Standard FP8 (8-bit float) staje się realny w nowych
 bibliotekach (PyTorch 2.1+). Flux LoRA zaleca FP8 minimum
 
 . Prawdopodobnie OneTrainer korzysta tu z triku: używa NF4 w weights i
 BF16 w aktywacjach, co ekwiwalentnie daje efektywność ~8-bit. W
 dokumencie flux było: „NF4 precision allows usage on lower VRAM but
 grid pattern artifact”. W przyszłości być może doczekamy się w OT opcji „WgradPrecision
 FP8”.

 • **Zrozumienie** **vs** **Memkorzystność:** Umiejętność doboru
 precyzji to cenna praktyczna umiejętność. Np. wiedza, że *przy*
 *dropout* *i* *DoRA* *w* *LoRA* *lepiej* *unikać* *FP16* *bo* *może*
 *generować* *NaNy* *–* *stąd* *BF16* *bezpieczniejsze*. Albo że
 *8-bit* *Adam* *obniża* *zużycie* *pamięci* *optymalizatora* *o*
 *~0.5x* *przy* *minimalnej* *różnicy* *w* *konwergencji*.

OneTrainer upraszcza te rzeczy, ale Ty, jako użytkownik zaawansowany,
potrafisz je kontrolować. Przykład: W training config w OT jest zapewne
flaga --mixed_precision (auto/FP16/BF16) – można wymusić np.
--mixed_precision no jakby FP32 potrzebne (czasem do debug).

W skrócie: Niższa precyzja = szybszy trening, mniejsze modele, ale
potencjalne problemy (instability). OneTrainer domyślnie stara się
używać rozsądnej mieszanej precyzji. W testach nie stwierdzono by to
powodowało istotne problemy.

**Różne** **typy** **LoRA** **(LoRA,** **LoHa,** **DoRA)**

Tradycyjna LoRA to nie jedyny sposób na efektywne dostrajanie modeli.
Wspominaliśmy: - **LoHa** **(Lycoris):** Rozszerzenie LoRA, które
wprowadza dodatkową macierz Hadamarda. Umożliwia to sieci nauczenie się
bardziej złożonych transformacji bez zwiększania wymiaru tak jak w
zwykłej LoRA. LoHa bywa lepsza do stylów artystycznych i drobnych
detali, potrafi uchwycić tekstury. OneTrainer obsługuje LoHa – wybierasz
*Type:* *LoHa* i dalej pracujesz analogicznie. W pliku LoRA wynikowym będzie po prostu
inne nazewnictwo tensorów (stąd np. Auto1111 wymaga skryptu do
załadowania LoHa albo wbudowanej obsługi – już jest mod na to). -
**DoRA** **(Weight** **Decomposition** **LoRA):** **Nowość** stworzona i
szybko zaimplementowana w OneTrainer (deweloper OT jest na bieżąco z
trendami). Pozwala trenować LoRA w dwóch częściach – modulując długość i kierunek
wektora wag osobno. Empiryczne wyniki (choć głównie z NLP) pokazały, że
to poprawia jakość adaptacji nawet do poziomu pełnego fine-tuningu. W OneTrainer aktywujesz to tickiem *Decompose* *Weights* w LoRA tab. W praktyce, DoRA może pomóc gdy LoRA normalna nie daje rady doskonale
odtworzyć czegoś. Z kolei wadą jest, że powstałe LoRA są mniej
standardowe – nie każde narzędzie je obsłuży (choć format pliku jest ten
sam, to wyjściowy adapter ma dwie pary matryc zamiast jednej – nie wiem
czy WebUI to zaaplikuje poprawnie bez aktualizacji). W repo comfyUI
widziałem zgłoszenia dot. wsparcia DoRA. - **LoRA** **tekstowe** **vs** **unetowe:** Standardowo LoRA dotyczy
U-Net (części generującej obraz). OneTrainer jednak (jak i kohya)
potrafi też trenować LoRA na text encoder. Taka LoRA tekstowa zmienia
embeddingi słów w CLIP – to inny efekt (coś jak textual inversion, ale w
parametry CLIP). OneTrainer daje możliwość *Layer*
*Preset:custom*iwskazaniawarstwtextencodera.JeślizaznaczyłbyśwLoRAtabnp.layeryCLIP-u,toLoRA
będzie także je obejmować. W praktyce mało kto tak robi (zwykle textual
inversion jest lepsze do tego), ale to możliwe. - **Kombinacje:**
OneTrainer umożliwia trenowanie jednocześnie LoRA + embedding, LoRA +
Dreambooth (regularization) – tak jakby mix technik. To dość unikalne.
Dla rekrutera: świadczy, że narzędzie ma elastyczność do eksperymentów
badawczych.

Z punktu widzenia kodu, obsługa LoHa i DoRA w OT to duży plus – nie
musisz zmieniać programu, by skorzystać z nowinek. Dla porównania, wiele
GUI (np. Kohya GUI) wymagało osobnych forków, by trenować LoHa.

Twoja świadomość typów LoRA pokazuje, że nie traktujesz LoRA jak czarnej
skrzynki, tylko rozumiesz, że to metoda *low-rank* *decomposition* i że
są jej wariacje. Możesz np. w CV czy rozmowie wspomnieć: „pracowałem z
implementacją LoRA i jej rozszerzeń (LoHa, DoRA) – potrafię je
zastosować i wiem kiedy są korzystne (np. DoRA do stabilizacji trudnych
treningów)”.

**Wybór** **trenowanych** **warstw** **modelu**

To dość niszowa, ale ciekawa opcja. Wspomnieliśmy w LoRA zakładce o
**Layer** **Preset** i możliwości custom. Dlaczego to jest ważne:

W pełnym fine-tuningu możemy np. *zamrozić* pewne warstwy i trenować
tylko inne (np. tylko ostatnie bloki UNet). W LoRA analogicznie – można
przyłożyć LoRA tylko do niektórych bloków. OneTrainer najnowszy pozwala
po prostu wpisać listę warstw (pewnie jako nazwy lub indeksy) w trybie
custom.

**Przykład** **zastosowania:** Trenujesz styl, który głównie wpływa na
kolory/tony, a nie strukturę obrazu – być może wystarczy LoRA na
warstwach odpowiedzialnych za drobne szczegóły i kolory (to mogłyby być
warstwy mid-level). Wtedy, zamiast marnować budżet parametrowy na
wszystkie warstwy, wybrałbyś tylko te kluczowe.

OneTrainer developer wspomniał, że standardowo dawno temu LoRA była
tylko na attention, potem wprowadzili *blocks* *=* *attn+MLP,* *all,*
*custom*. To czyni narzędzie dość unikalnym – bo np. w Kohya GUI musiałeś
kombinować z parametrami network_dim i skryptami, a tu masz gotowe.

Dlaczego rekruter miałby się tym przejąć? Bo pokazuje to *głębokie*
*zrozumienie* *architektury* *Stable* *Diffusion*. Aby sensownie wybrać
warstwy, musisz wiedzieć jak U-Net jest zbudowany (bloki down, mid, up,
attn layers, MLP in feed-forward etc.). To dość zaawansowana wiedza
inżynierska.

Możesz np. powiedzieć: „Aby uniknąć efektu 'purple bleed' w LoRA
trenowanej dla modelu SD2.1, ograniczyłem LoRę do warstw attention i
pierwszych warstw decodera – to usunęło problem nadpisywania palety
kolorów.” Taka analiza robi wrażenie i narzędzie Ci to umożliwiło.

Podsumowując, OneTrainer to **rozbudowane** **środowisko** i Ty
poznałeś/łaś jego zakamarki: ofloading, precision, LoRA types, layer
control. Te funkcje są zdecydowanie zaawansowane i wykraczają poza
„kliknij i czekaj”. Ich opanowanie dowodzi praktycznych umiejętności w
trenowaniu modeli AI i rozumienia mechanizmów uczenia.

**Praktyczne** **porady** **i** **najlepsze** **praktyki**

Na zakończenie, kilka dodatkowych porad z doświadczeń użytkowników
OneTrainer i ogólnych zasad:

 • **Zawsze** **monitoruj** **przebieg** **treningu:** Nie zostawiaj
 modelu na wiele godzin bez nadzoru, zwłaszcza na początku. Obserwuj
 loss – jeżeli nagle staje się NaN lub rośnie do nieskończoności,
 przerwij – coś poszło nie tak (zbyt duży LR, niestabilna precyzja
 FP16). OneTrainer co prawda stara się temu zapobiec (Mixed Precision,
 grad clipping), ale czujność popłaca.

 • **Regularnie** **korzystaj** **z** **próbek** **i** **walidacji:**
 Jak już wspomniano, sample images co pewien czas to cenny feedback.
 Walidacja loss też się przydaje. Choć czasem „twoje oczy są najlepszą
 miarą” – generuj obrazy testowe w docelowym UI, bo ostatecznie liczy
 się czy efekt wizualny Ci odpowiada, nie minimalny loss.

 • **Nie** **przeuczaj** **LoRA:** Częsty błąd – zbyt długo trenowana
 LoRA zaczyna „przepisywać” obrazy treningowe w pamięci i traci
 zdolność generalizacji. Objawy: generowane obrazy wyglądają bardzo
 podobnie do treningowych (overfitting) albo model reaguje tylko na
 bardzo specyficzny prompt. Recepty:

 • Używaj regularyzacji: dropout (w LoRA), augmentacji, prior images
 (przy DreamBooth).

 • Zastosuj Early Stopping manualnie – gdy widzisz na próbkach, że już
 jest dobrze, zakończ trening. LoRA często osiąga optimum szybciej niż
 by wskazywał spadek loss (loss może dalej spadać minimalnie, ale
 obrazki już się nie poprawiają, a nawet pogarszają – bo np.
 kolorystyka staje się monotonna).

 • **Porównuj** **różne** **podejścia:** OneTrainer sprawia, że test
 A/B jest łatwy – np. możesz skopiować config, zmienić jedną rzecz (np.
 LoRA rank 8 vs 16) i zrobić dwa treningi. Potem odpalić generację z
 obu LoR i porównać. Wyciągaj wnioski i zapisuj je (Lessons Learned
 wiki OT zawiera sporo takich wniosków od użytkowników).

 • **Wykorzystuj** **Dev** **Corner:** Oficjalne wiki OneTrainer ma
 sekcję Dev Corner, gdzie znajdują się wskazówki dla programistów i zaawansowanych
 użytkowników. Można tam znaleźć np. omówienie struktury projektu, co ułatwi Ci ewentualne debugowanie czy dodawanie

 własnych funkcji. Dla rekrutera może mieć znaczenie, że potrafisz
 czytać kod narzędzia i ewentualnie go modyfikować – w OT to możliwe,
 bo kod jest czysty (tkinter, PyTorch, Diffusers).

 • **Common** **mistakes** **(częste** **błędy):** Wiki OT ma sekcję
 poświęconą typowym potknięciom użytkowników migracji z kohya. Np. ludzie zapominają, że w OneTrainer workspace musi być ustawiony
 poprawnie, inaczej generowane sample nie zapiszą się tam gdzie myślą;
 albo że przy kontynuacji treningu z backup *nie* *należy* *zmieniać*
 *nic* *w* *konfiguracji*, bo to może dać niespójność. Zalecenie: jeśli
 wznawiasz trening, **ładuj** **ten** **sam** **plik** **preset,**
 **co** **użyty** **na** **początku**, i tylko zaznacz „Continue from
 last backup” – nie przekonfiguruj opcji, bo backup zawiera
 optymalizator, który np. jest w środku cyklu scheduler itp.

 • **Aktualizacje** **OneTrainer:** Narzędzie jest rozwijane aktywnie.
 Sprawdzaj GitHuba co pewien czas – często pojawiają się usprawnienia
 UI, nowe optymalizatory (np. niedawno dodano *Prodigy*, *Lion*),
 wsparcie nowych architektur. Aktualizację zrobisz łatwo: git pull i
 update.bat (instaluje nowe require).
 Zadbaj o kompatybilność – czasem nowe wersje zmieniają format
 presetów, więc warto też czytać release notes.

 • **Społeczność:** Dołącz na **Discord** **OneTrainer** – link
 znajdziesz w repo. Tam możesz podzielić się wynikami, zapytać o rady, a nawet
 współtworzyć rozwój (autor jest otwarty na feedback). Bycie aktywnym w
 takiej społeczności pokazuje pasję i bycie na bieżąco.

 • **Eksperymentuj** **i** **dokumentuj:** Prowadź notatki z
 eksperymentów (np. różne parametry LoRA i ich efekt). To pozwoli Ci
 budować intuicję. Wiki *Info,* *Guides* *and* *Lessons* *Learned*
 zawiera doświadczenia innych – np. jaki dropout działał dla jakiego
 typu stylu, czy warto trenować text encoder itp. Warto się inspirować,
 ale każde nowe zadanie może być inne – Twoja dociekliwość by testować
 różne rozwiązania jest atutem.

Na koniec: Posiadając repozytorium wiedzy tak przygotowane (w formie
tego README), dysponujesz świetnym materiałem, by pokazać rekruterowi.
Prezentuje on, że:

 • Znasz narzędzia AI **w** **praktyce** – potrafisz nie tylko ich
 użyć, ale i skonfigurować głęboko pod wymagania.

 • Rozumiesz **działanie** **LoRA** **i** **Stable** **Diffusion** – co
 widać po objaśnieniach parametrów takich jak rank, alpha, dropout,
 maski, bucketing.

 • Potrafisz **automatyzować** **i** **programować** – znajomość trybu
 CLI, skryptów, integracji z innymi systemami to ważna umiejętność
 inżynierska.

 • Masz **świadomość** **najnowszych** **trendów** – wspomnienie flux,
 DoRA, LoHa pokazuje, że śledzisz dynamiczny rozwój technologii
 generatywnych (2024/2025 to właśnie czasy SDXL, nowe optymalizatory,
 flux/diffusers eksperymenty).

 • Jesteś **dokładny** **i** **zorganizowany** – struktura dokumentu,
 cytowanie źródeł z oficjalnej dokumentacji, porządek sekcji, wszystko to odzwierciedla umiejętność tworzenia
 czytelnej dokumentacji technicznej, co również jest cenione.

Powodzenia w dalszych projektach z OneTrainer – niech ten przewodnik
służy zarówno Tobie, jak i każdemu, kto chce wejść w świat trenowania
własnych modeli generatywnych!
