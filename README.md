# Multi-label classificatie van koraalrifcondities met deep learning

## Over dit project

Koraalriffen staan wereldwijd onder druk. Het handmatig beoordelen van onderwaterbeelden kost tot 80 minuten per 100 meter transect. Dit project onderzoekt of deep learning die classificatie kan automatiseren.

Een CNN from scratch en drie transfer learning modellen (EfficientNet-B7, ResNet-101, DenseNet-201) worden vergeleken op multi-label classificatie van vier koraalcondities: healthy, compromised, dead en rubble. Het beste model (EfficientNet-B7, frozen backbone, lr=1e-3, 512px) haalt na drempeloptimalisatie per label een macro F1 van 0.58 op de testset.

## Dataset

**Koh Tao Coral Condition Dataset** van Shao et al. (2024, 2025)

- 23.924 beeldpatches na deduplicatie (origineel 23.965).
- Resolutie: 512 x 512 pixels.
- 4 conditielabels: healthy, compromised, dead, rubble.
- Verzameld op 15 duiklocaties rond Koh Tao, Thailand.
- Geannoteerd door experts in mariene ecologie.

Bron: [GitHub](https://github.com/XL-SHAO/CoralConditionDataset) en [Google Drive](https://drive.google.com/drive/folders/1yjvVGSXuFRcO3b0SehyeAHzMtCHhI6S1)

## Repositorystructuur

```
coral_multilabel_classification.ipynb   # Hoofdnotebook (rapport + code)
README.md                               # Dit bestand
requirements.txt                        # Python dependencies
```

## Resultaten

| Model | Macro F1 | Match ratio | Set |
|-------|----------|-------------|-----|
| Null model | 0.21 | 0.33 | test |
| LR baseline | 0.45 | 0.10 | test |
| CNN from scratch | 0.50 | 0.12 | test |
| EfficientNet-B7 (getuned) | 0.58 | 0.26 | test |

Per-label resultaten (testset, geoptimaliseerde drempels):

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| healthy | 0.80 | 0.92 | 0.86 |
| compromised | 0.39 | 0.58 | 0.47 |
| dead | 0.46 | 0.82 | 0.59 |
| rubble | 0.29 | 0.69 | 0.40 |

## Uitvoeren

### Google Colab (aanbevolen)

1. Upload het notebook naar Google Colab
2. Stel de runtime in op GPU
3. Voer alle cellen uit van boven naar beneden

De dataset wordt automatisch gedownload vanuit GitHub en Google Drive.

Link naar Google Colab: https://colab.research.google.com/drive/1fjBjI9kLr6ZSR1dRz-J_9pOIEjhtPvr3?usp=sharing

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

- Python 3.12
- PyTorch 2.10.0, Torchvision 0.25.0
- GPU: NVIDIA RTX PRO 6000 (95 GB) via Google Colab
- Mixed precision (AMP) en channels_last geheugenformaat
- Deterministic mode ingeschakeld (seed=42)

## Aanpak

1. **Data voorbereiding**: deduplicatie, group-aware split (70/10/20), EDA op trainingsset
2. **Baselines**: null model en logistic regression op kleurkenmerken
3. **Iteratie 1**: CNN from scratch (4 convolutielagen)
4. **Iteratie 2**: drie transfer learning modellen met frozen backbone
5. **Ablations**: learning rate (1e-4 vs 1e-3) en resolutie (512 vs 224) apart gevarieerd
6. **Drempeloptimalisatie**: per label op de validatieset
7. **Evaluatie**: eenmalig op de testset
8. **Interpreteerbaarheid**: GradCAM per conditielabel

## Auteur

Rainesh Rewat

## Referenties

Shao, X. et al. (2024). Deep Learning for Multilabel Classification of Coral Reef Conditions in the Indo-Pacific. *Aquatic Conservation*, 34(9), e4241.

Shao, X. et al. (2025). Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring. *arXiv:2503.23012*.
