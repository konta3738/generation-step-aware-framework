# Generation-Step-Aware Framework for Cross-Modal Representation and Control in Multilingual Speech–Text Models

## Overview

This repository contains the official implementation of our work on **generation-step-aware diagnostic framework** for multilingual and multimodal models.

We introduce a framework to:
- Identify language-selective neurons at different generation steps
- Decompose neuron roles into **representation** and **control**
- Evaluate cross-modal alignment between speech and text
- Perform **causal interventions** to steer model behavior across modalities

The implementation supports:
- Speech-to-text (ASR / S2T)
- Text-to-text translation
- Multilingual analysis across languages

---

## 🔗 Dependencies on Prior Work

This code builds upon the following repositories:

- **Cuadros et al. (2022)** – Self-conditioning Pre-Trained Language Models  
  https://github.com/apple/ml-selfcond  

- **Kojima et al. (2024)** – On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons  
  https://github.com/kojima-takeshi188/lang_neuron  

We thank the authors for making their code publicly available.

---

## ⚖️ License

- Portions of this code are derived from third-party sources and retain their original licenses:
  - Apple Inc. (2022)
  - Kojima et al. (2024)

- All original license notices are preserved in the corresponding files.

- Modifications and newly added code in this repository are:

Copyright (c) 2026 Toshiki Nakai

👉 This repository is released under **MIT License**, *except for parts that inherit other licenses*.

---

## 📦 Contents

This repository includes:

- Activation recording pipeline
- AP-based neuron identification
- Top / bottom-k neuron extraction
- Cross-lingual & cross-modal overlap analysis
- Distribution visualization tools
- Intervention-based generation experiments

---

## 📊 Data

### Speech Data

- Speech data is **synthetically generated** (XTTS-based)
- Audio files are **not distributed** to avoid potential identification risks

However:
- ✅ Data generation scripts are provided  (xtts_vc.py)

---

## ⚙️ Requirements

- Python 3.9+
- PyTorch
- HuggingFace Transformers
- datasets
- pandas, numpy, matplotlib
- Coqui TTS (for speech generation)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Minimal Usage

### 1. Activation Recording

```bash
bash ./main_prod_env.sh seamless-m4t-v2-large compute_responses Speech de_speech_VC 1000 on_p50 text_decoder_expertise_limited_1000_both s2t_translation 0 deu eng 16000
```

---

### 2. AP-based Neuron Identification

```bash
bash ./main_prod_env.sh seamless-m4t-v2-large compute_expertise Speech de_speech_VC 2000 on_p50 expertise_limited_2000_both s2t_translation "" deu "" 16000
```

---

### 3. Top / Bottom-k Extraction

```bash
bash ./main_prod_env.sh seamless-m4t-v2-large limit_expertise Speech "" 1000 on_p50 expertise_limited_1000_both s2t_translation "" "" "" 16000
```

---

### 4. Plotting

#### Overlap Heatmaps

```bash
python make_plots.py \
  --do-overlap \
  --root-s2t "./set_appropriate_path_2/Speech/s2t_translation/seamless-m4t-v2-large/sense" \
  --root-t2t "./set_appropriate_path_2/Speech/t2t_translation/seamless-m4t-v2-large/sense" \
  --overlap-out "./overlap_heatmaps"
```

#### Distribution Plots

```bash
python make_plots.py \
  --do-stacked \
  --sense-roots \
    "./set_appropriate_path_2/Speech/s2t_translation/seamless-m4t-v2-large/sense" \
    "./set_appropriate_path_2/Speech/t2t_translation/seamless-m4t-v2-large/sense" \
  --figure-root "./figure"
```

---

### 5. Intervention-Based Generation

Run generation with selected neuron sets (top / bottom neurons).

Example scripts are provided in the repository (generate_seq_lang.py).

---

## 🔬 Reproducibility

- Experiments are configurable across:
  - Languages (de, es, fr, zh, ja, etc.)
  - Tasks (ASR, S2T, T2T)
  - Neuron selection sizes (top-k, bottom-k)

- Default settings match those used in the paper.

---

## 📌 Notes

- Model checkpoints (e.g., SeamlessM4T v2 large) are downloaded via HuggingFace
- External datasets are required and not redistributed (you can apply xtts_vc.py on the public FLEURS dataset after having one voice data to condition the voice on.)
- Paths in scripts may need to be adjusted depending on your environment

---

## 📄 Citation

If you use this code, data, or experimental setup, please cite:

### Our work

```bibtex
@article{nakai2026gsad,
  title={Generation-Step-Aware Neuron Analysis for Multilingual and Multimodal Models},
  author={Nakai, Toshiki},
  year={2026}
}
```

---

### Datasets

**FLEURS (Conneau et al., 2023)**

```bibtex
@inproceedings{conneau2023fleurs,
  title={FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and others},
  booktitle={IEEE Spoken Language Technology Workshop (SLT)},
  year={2022},
  pages={798--805},
  doi={10.1109/SLT54892.2023.10023141}
}
```

---

### Models

**SeamlessM4T (Seamless Communication et al., 2023)**

```bibtex
@misc{communication2023seamlessmultilingualexpressivestreaming,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Seamless Communication and Lo{\"i}c Barrault and Yu-An Chung and others},
  year={2023},
  eprint={2312.05187},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2312.05187}
}
```

**XTTS (Casanova et al., 2024)**

```bibtex
@inproceedings{casanova24_interspeech,
  title={{XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model}},
  author={Casanova, Edresson and Davis, Kelly and G{\"o}lge, Eren and G{\"o}knar, G{\"o}rkem and Gulea, Iulian and Hart, Logan and Aljafari, Aya and Meyer, Joshua and Morais, Reuben and Olayemi, Samuel and Weber, Julian},
  booktitle={{Interspeech 2024}},
  year={2024},
  pages={4978--4982},
  doi={10.21437/Interspeech.2024-2016}
}
```

---

### Prior Work

**Self-Conditioning (Cuadros et al., ICML 2022)**

```bibtex
@inproceedings{pmlr-v162-cuadros22a,
  title={Self-conditioning Pre-Trained Language Models},
  author={Cuadros, Xavier Suau and Zappella, Luca and Apostoloff, Nicholas},
  booktitle={Proceedings of the 39th International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={162},
  pages={4455--4473},
  year={2022},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v162/cuadros22a.html}
}
```

**Language-Specific Neurons (Kojima et al., NAACL 2024)**

```bibtex
@inproceedings{kojima-etal-2024-multilingual,
  title={On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons},
  author={Kojima, Takeshi and Okimura, Itsuki and Iwasawa, Yusuke and Yanaka, Hitomi and Matsuo, Yutaka},
  booktitle={Proceedings of NAACL-HLT 2024},
  year={2024},
  pages={6919--6971},
  doi={10.18653/v1/2024.naacl-long.384}
}
```

---

## 🙏 Acknowledgements

This work builds upon:
- Cuadros et al. (2022)
- Kojima et al. (2024)

---

## 📬 Contact

Toshiki Nakai  

toshiki3738 (at) gmail.com
