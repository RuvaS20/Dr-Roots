# 🌿 Doctor Roots

**A WhatsApp bot that identifies indigenous African medicinal plants from a photo — and tells you what the research actually says about them.**

Send a picture of a leaf to a WhatsApp number. A MobileNet-based classifier running on a Flask server identifies the species and replies with its Shona name, physical description, documented medicinal uses, preparation methods, IUCN conservation status, and links to peer-reviewed sources.

🔗 **Live:** [dr-roots-bff0f37dc742.herokuapp.com](https://dr-roots-bff0f37dc742.herokuapp.com/)
📱 **Try it:** WhatsApp `+1 415 523 8886`, send `join silly-degree`

---

## Why this exists

The WHO Regional Office for Africa reports that **70–80% of the region uses traditional medicine**, with herbal treatments the most popular form [[1]](#references). In south-central Zimbabwe alone, an ethnobotanical review documented **93 plant species used across 100 medicinal applications** [[2]](#references) — knowledge that is transmitted orally and unevenly.

Misidentification is the failure mode that matters. A global DNA-barcoding analysis of **5,957 commercial herbal products across 37 countries** found **27% were adulterated** relative to their labelled species, with regional rates from 19% to 79% [[3]](#references). When the wrong plant is used, the consequences are real: *Aloe vera* taken orally can cause electrolyte depletion and interacts with digoxin and furosemide [[4]](#references) — safety information most users never see.

Doctor Roots targets that gap with the channel people already have. Just over half of mobile connections in Sub-Saharan Africa are smartphones [[5]](#references), and WhatsApp requires no app install, no account, and works on low-end devices — so identification runs server-side and the phone only sends a photo.

---

## The seven species

| # | Scientific name | Common name | Shona name | IUCN status |
|---|---|---|---|---|
| 1 | *Aloe barbadensis* | Aloe vera | Gavakava | Least Concern |
| 2 | *Catharanthus roseus* | Madagascar Periwinkle | Chirindamatongo | Not assessed |
| 3 | *Citrus limon* | Lemon | Mulemoni | Not assessed |
| 4 | *Mangifera indica* | Mango | Mumango | Not assessed |
| 5 | *Moringa oleifera* | Moringa | Moringa | Not assessed |
| 6 | *Psidium guajava* | Guava | Mugwavha | Not assessed |
| 7 | *Zingiber officinale* | Ginger | Tsangamidzi | Not assessed |

 Per-species profiles, safety precautions, and citations live in [plant_data.json](plant_data.json).

---

## How it works

```text
WhatsApp photo → Twilio webhook → Flask (/webhook) → download media
                                        ↓
                       resize 224×224, scale to [0,1], float32
                                        ↓
                       TFLite interpreter → softmax over 7 classes
                                        ↓
              confidence ≥ 0.70 ? → plant_data.json lookup → reply
                                  ↘ else → "try another image"
```

The conversation is a small state machine (`menu` → `default` → `selecting_plant`) held in memory per phone number, implemented in [app.py:186-336](app.py#L186-L336). Predictions below the 0.70 confidence threshold are refused.

### Model

| | |
|---|---|
| Base | MobileNet (ImageNet weights, frozen) [[6]](#references) |
| Head | GlobalAveragePooling → Dropout 0.5 → Dense 128 (ReLU, L2 0.01) → Dropout 0.3 → Dense 7 (softmax) |
| Input | 224×224×3, rescaled to [0,1] |
| Optimiser | Adam, lr 0.001, categorical cross-entropy |
| Training | 10 epochs, batch size 64 |
| Serving | TFLite, 13 MB, CPU-only |

MobileNet was chosen for its depthwise-separable convolutions, which trade a small amount of accuracy for a large reduction in parameters and latency [[6]](#references).

### Results

| Metric | Value |
|---|---|
| Test accuracy | **95.8%** |
| Validation accuracy (final epoch) | 96.5% |
| Test loss | 0.54 |

Held-out test split, evaluated in [dr_roots.ipynb](dr_roots.ipynb). Validation accuracy exceeded 93% by epoch 2 and plateaued around 96% — consistent with published transfer-learning results on medicinal leaf datasets, which typically report 88–99% depending on class count and image quality.

### Data preparation

Images were collected per species and processed with the scripts in this repo before training:

- [resize_images.py](resize_images.py) — LANCZOS resize to 224×224
- [remove_background.py](remove_background.py) — Otsu thresholding + morphological cleanup, largest-contour leaf extraction
- [augment_images.py](augment_images.py) — Albumentations pipeline (rotation, flips, noise, blur, elastic/optical distortion), 5 augmentations per source image

Background removal matters here: leaf photos taken in the field carry soil, hands, and other foliage that a small model will happily learn instead of the leaf.

---

## Repository layout

| File | Purpose |
|---|---|
| [app.py](app.py) | Flask server, Twilio webhook, inference, conversation state |
| [dr_roots.py](dr_roots.py) / [dr_roots.ipynb](dr_roots.ipynb) | Training, evaluation, TFLite conversion ([Colab](https://colab.research.google.com/drive/1lVsSBc4DcITlxcU_-VavLL0wJ1lONRXA?usp=sharing)) |
| [plant_data.json](plant_data.json) | Species profiles, uses, safety notes, citations |
| [class_mapping.json](class_mapping.json) | Class index → scientific name |
| `dr_roots_model.h5` / `dr_roots_model.tflite` | Trained Keras model and TFLite export |
| [resize_images.py](resize_images.py), [remove_background.py](remove_background.py), [augment_images.py](augment_images.py) | Dataset preprocessing |
| [Procfile](Procfile), [runtime.txt](runtime.txt), [requirements.txt](requirements.txt) | Heroku deployment config |

---

## Running locally

```bash
git clone https://github.com/RuvaS20/Dr-Roots.git
cd Dr-Roots
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your [Twilio](https://console.twilio.com/) credentials:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=whatsapp:+14155238886
```

Then run the server and expose it so Twilio can reach it:

```bash
python app.py                    # http://localhost:5000
ngrok http 5000                  # in a second terminal
```

Set the ngrok HTTPS URL + `/webhook` as the **"When a message comes in"** endpoint in the Twilio WhatsApp Sandbox console, then message the sandbox number.

**Deploying:** the repo is Heroku-ready — `gunicorn app:app` on Python 3.10.12. Set the three Twilio variables as config vars and point the Twilio webhook at `https://<your-app>.herokuapp.com/webhook`.

**Retraining:** open [dr_roots.ipynb](dr_roots.ipynb) in Colab with a dataset at `MyDrive/medicinal_plants/data/<Species>/{Train,Validation,Test}/`. The notebook trains, evaluates, writes `class_mapping.json`, and converts to TFLite.

---

## Known limitations

- **Name mismatch between the model and the plant database.** `class_mapping.json` emits `"Citrus Limon"`, `"Moringa oleifera"`, and `"Zingiber officianale"`, while [plant_data.json](plant_data.json) keys them as `"Citrus limon"`, `"Moringa oleifera Lour"`, and `"Zingiber officinale Roscoe"`. Image predictions for those three species currently fall through to *"Plant not found in database"*. The menu-driven lookup path is unaffected.
- **Conversation state is in-process.** `user_states` is a plain dict, so it resets on restart and does not survive multiple dynos.
- **Seven classes, closed set.** Any photo is forced into one of seven species; the 0.70 threshold is the only guard against out-of-distribution input. There is no "not a plant" or "unknown species" class.
- **`imghdr` is deprecated** and removed in Python 3.13; the runtime is pinned to 3.10.

---

## Medical disclaimer

Doctor Roots is an **educational and research tool**. It is not a diagnostic device and does not provide medical advice. Plant identification from a single photograph is inherently uncertain, and many medicinal plants have serious contraindications, drug interactions, and toxic look-alikes. Do not use this tool to decide whether to consume any plant. Consult a qualified healthcare provider or trained herbalist.

---

## References

1. World Health Organization, Regional Office for Africa. *Traditional Medicine.* https://www.afro.who.int/health-topics/traditional-medicine
2. Maroyi, A. (2013). Traditional use of medicinal plants in south-central Zimbabwe: review and perspectives. *Journal of Ethnobiology and Ethnomedicine*, 9, 31. https://doi.org/10.1186/1746-4269-9-31
3. Ichim, M. C. (2019). The DNA-based authentication of commercial herbal products reveals their globally widespread adulteration. *Frontiers in Pharmacology*, 10, 1227. https://doi.org/10.3389/fphar.2019.01227
4. Surjushe, A., Vasani, R., & Saple, D. G. (2008). Aloe vera: A short review. *Indian Journal of Dermatology*, 53(4), 163–166. https://doi.org/10.4103/0019-5154.44785
5. GSMA. (2024). *The Mobile Economy Sub-Saharan Africa 2024.* https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/sub-saharan-africa-2024/
6. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv:1704.04861.* https://arxiv.org/abs/1704.04861

Per-species pharmacological references are listed in [plant_data.json](plant_data.json) and surfaced to users in every bot reply.

---

## License

GNU General Public License (GPL)

## Contact

**Ruvarashe Sadya** — [ruvarashe.sadya@gmail.com](mailto:ruvarashe.sadya@gmail.com) · [github.com/RuvaS20](https://github.com/RuvaS20)

Contributions to the plant knowledge base — especially additional species, vernacular names, and sourced safety information — are welcome.
