# ⚽ Fotbolls-videoanalys med YOLOv8

Automatisk videoanalys av fotbollsmatcher. Ta en matchvideo (från **XbotGo
Falcon AI-kamera** eller **YouTube**) och detektera automatiskt spelare och boll
med **YOLOv8**, spåra rörelser över tid och generera enkel statistik
(heatmaps, possession-estimering) samt textbaserade insikter till tränaren.

Projektet kan köras **lokalt** eller i **Google Colab** (GPU rekommenderas).

---

## 📁 Projektstruktur

```
football-video-analysis/
├── requirements.txt          # Python-beroenden
├── config/
│   └── config.yaml           # All konfiguration (modell, video, lag, output)
├── src/                      # Källkod (importerbart paket)
│   ├── video.py              # Läs/skriv video + YouTube-nedladdning
│   ├── detection.py          # YOLOv8-detektering + spårning (ByteTrack)
│   ├── teams.py              # Enkel lag-klassificering via tröjfärg
│   ├── stats.py              # Heatmaps, possession, löpsträcka, insikter
│   ├── annotate.py           # Ritar boxar/ID/etiketter på video
│   └── pipeline.py           # Orkestrerar hela kedjan
├── scripts/
│   └── analyze_match.py      # CLI-ingång
├── notebooks/
│   └── colab_football_analysis.ipynb   # Kör i Google Colab
├── data/                     # Indata-videor (gitignorerade)
└── output/                   # Genererad output (gitignorerad)
```

---

## 🚀 Kom igång (lokalt)

Kräver Python 3.9+.

```bash
cd football-video-analysis

# 1. Skapa virtuell miljö (rekommenderas)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Installera beroenden (första gången laddas YOLO-vikter ner automatiskt)
pip install -r requirements.txt

# 3. Analysera en lokal video
python scripts/analyze_match.py --video data/match.mp4

# ...eller ladda ner direkt från YouTube
python scripts/analyze_match.py --youtube "https://youtu.be/DIN_LANK"
```

### Snabbtest (rekommenderas första gången)

Analysera bara några hundra bildrutor med den minsta modellen på CPU:

```bash
python scripts/analyze_match.py --video data/match.mp4 --max-frames 300
```

---

## ⚙️ Vanliga flaggor

| Flagga            | Beskrivning                                        |
|-------------------|----------------------------------------------------|
| `--video`         | Sökväg till lokal videofil                         |
| `--youtube`       | YouTube-URL att ladda ner och analysera            |
| `--weights`       | YOLO-modell (`yolov8n/s/m/l/x.pt`)                  |
| `--conf`          | Konfidenströskel för detektering (0–1)             |
| `--device`        | `''` (auto), `cpu` eller `0` (GPU)                 |
| `--max-frames`    | Begränsa antal rutor (0 = hela videon)             |
| `--frame-stride`  | Analysera var N:te ruta (snabbare)                 |
| `--enable-teams`  | Slå på lag-detektering via tröjfärg                |
| `--no-video`      | Hoppa över att spara annoterad video               |

All konfiguration kan även sättas permanent i `config/config.yaml`.

---

## 📊 Vad du får ut

Efter en körning skapas i `output/`:

- **`*_annotated.mp4`** – videon med boxar, spelar-ID och (valfritt) lagfärger.
- **`*_heatmap_players.png`** – heatmap över spelarpositioner.
- **`*_heatmap_ball.png`** – heatmap över bollen.
- **`*_stats.json`** – possession, antal spårade ID, löpsträcka per spelare m.m.
- Textbaserade **insikter till tränaren** skrivs ut i terminalen.

---

## 🎽 Lag-detektering och possession

Possession-estimering kräver att spelare kan tilldelas ett lag. Den första
versionen använder en enkel heuristik: dominerande **tröjfärg** i övre delen av
varje spelares box (HSV-intervall).

1. Aktivera med `--enable-teams` eller `teams.enable: true` i configen.
2. Justera HSV-intervallen (`team_a_hsv_*`, `team_b_hsv_*`) efter lagens färger.

> Possession beräknas som andelen bildrutor där en spelare från respektive lag
> är närmast bollen. Det är en approximation – bra som riktvärde, inte exakt
> matchdata.

---

## ☁️ Google Colab

Öppna `notebooks/colab_football_analysis.ipynb` i Colab, välj **Runtime → GPU**,
och kör cellerna uppifrån och ner. Notebooken installerar beroenden, låter dig
ladda upp eller länka en video, kör analysen och visar resultaten inline.

---

## 🧭 Roadmap / nästa steg

- [ ] Homografi/planmappning så heatmaps blir i riktiga plankoordinater (meter).
- [ ] Robustare lag-klustring (k-means på färger) i stället för fasta HSV-intervall.
- [ ] Kalibrering av löpsträcka från pixlar till meter.
- [ ] Passningsdetektering och skottkarta.
- [ ] Export av rapport (PDF/HTML) till tränaren.

---

## 📝 Licens / användning

Internt verktyg för matchanalys. Respektera upphovsrätt för videomaterial du
laddar ner (t.ex. från YouTube).
