# Multi-label classificatie van koraalrifcondities

Multi-label classificatie van koraalrifcondities (healthy, compromised, dead, rubble) op de Koh Tao Coral Condition Dataset, met een vergelijking tussen een CNN from scratch en transfer learning met ImageNet-pretrained backbones (ResNet-101, DenseNet-201, EfficientNet-B7).

**Auteur:** Rainesh Rewat

## Repositorystructuur

```
.
├── coral_multilabel_classification.ipynb   Hoofdnotebook (rapport + code)
├── README.md                               Dit bestand
├── requirements.txt                        Python dependencies
└── .gitignore                              Git ignore-regels
```

Het notebook bevat zowel de methodologische beschrijving als de uitvoerbare code en is het enige deliverable dat daadwerkelijk uitgevoerd hoeft te worden.

## Dataset

De Koh Tao Coral Condition Dataset (Shao et al., 2024) bevat gelabelde 512×512 patches uit onderwateropnames op Koh Tao, Thailand. Vier conditielabels zijn gebruikt: healthy, compromised, dead en rubble.

Download de dataset via Google Drive:
https://drive.google.com/drive/folders/1-D5oFKGHvJv9Yf7oIG5m3OvmIplIk6OX?usp=sharing

Plaats de gedownloade bestanden zoals aangegeven in sectie 5 van het notebook. Bij gebruik van Google Colab kan de dataset direct vanuit Google Drive worden gemount.

Originele bron: Shao, X., Chen, H., Magson, K., Wang, J., Song, J., Chen, J., Sasaki, J. (2024). Deep Learning for Multilabel Classification of Coral Reef Conditions in the Indo-Pacific Using Underwater Photo Transect Method. *Aquatic Conservation: Marine and Freshwater Ecosystems*, 34(9), e4241.

## Python versie en omgeving

- Python 3.12 of hoger
- CUDA-capabele GPU vereist (T4 of beter)
- BF16-support wordt automatisch gedetecteerd (is_bf16_supported), anders wordt FP16 of FP32 gebruikt

## Installatie

### Optie 1 — Google Colab (aanbevolen)

Het notebook is ontwikkeld en getest in Google Colab en kan rechtstreeks worden geopend:

https://colab.research.google.com/drive/1sFf7l3w44vlk7SkQXHl7-zKH-zPrCPJp?usp=sharing

Stappen:
1. Open de bovenstaande link
2. Zet de runtime op GPU: Runtime → Change runtime type → T4 GPU
3. Voer alle cellen uit (Runtime → Run all)

Colab heeft de meeste dependencies al voorgeïnstalleerd. De `imagehash` library wordt automatisch geïnstalleerd in de eerste cellen van het notebook.

### Optie 2 — Lokale installatie

```bash
git clone <repo-url>
cd coral-condition-classification
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
jupyter notebook coral_multilabel_classification.ipynb
```
### Lokaal

1. Maak een virtuele omgeving aan:
   ```
   python -m venv venv
   source venv/bin/activate
   ```
2. Installeer dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Zorg voor een GPU met minimaal 16 GB VRAM
4. Voer het notebook uit in Jupyter:
   ```
   jupyter notebook coral_multilabel_classification_v1.ipynb
   ```

## Trainingsomgeving

- Python 3.12.13
- PyTorch 2.10.0, Torchvision 0.25.0
- GPU: NVIDIA RTX PRO 6000 (95 GB) via Google Colab
- Mixed precision (AMP) en channels_last geheugenformaat

## Experimenten uitvoeren

Het notebook is zo opgebouwd dat het van begin tot eind uitvoerbaar is zonder handmatige tussenstappen. Voer de cellen in volgorde uit:

1. Setup en configuratie (secties 5–6)
2. Datalaad en split (secties 7–10)
3. EDA op de trainset (secties 11–12)
4. Iteratie 1: CNN from scratch (secties 13–14)
5. Iteratie 2: Vergelijking transfer learning modellen (secties 15–16)
6. Iteratie 3: Ablationstudies op EfficientNet-B7 (secties 17–19)
7. Robuustheidstest, drempeloptimalisatie en finale testevaluatie (secties 20–22)
8. GradCAM op misclassificaties (sectie 23)

Reproduceerbaarheid is geborgd via vaste seeds, cuDNN deterministic mode en group-aware splitsing op locatie.

### Kerncijfers

**Vergelijking op testset (macro F1)**

| Model | Test macro F1 | Match ratio |
|---|---|---|
| Null model | 0.2120 | 0.3344 |
| Logistic regression baseline | 0.4557 | 0.0989 |
| CNN from scratch | 0.5123 | 0.1584 |
| EfficientNet-B7 (drempel 0.50) | 0.6604 | 0.3515 |
| EfficientNet-B7 (tuned drempels) | 0.6595 | 0.3789 |

**Beste setting:** EfficientNet-B7, learning rate 1e-3 op de classificatielaag, resolutie 512×512, met augmentatie.

## Evaluatie

De primaire metriek is macro F1, zodat alle vier de labels even zwaar meewegen ondanks de onbalans tussen healthy (meest voorkomend) en rubble (zeldzaamst). Secundaire metrieken zijn match ratio, per-label precision en recall, en per-label F1. De testset wordt alleen ingezet voor de finale evaluatie. Modelselectie en drempeloptimalisatie vinden uitsluitend op de validatieset plaats.

## Licentie

MIT License.
