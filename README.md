# NAION-Risk-Analyzer
Deep Learning (U-Net) tool for automated optic nerve head segmentation and NAION risk profiling via anatomical biomarkers.

# NAION-Risk: AI-Driven Ocular Diagnostic Support

An end-to-end computer vision application designed to quantify anatomical and vascular biomarkers associated with **Non-Arteritic Anterior Ischemic Optic Neuropathy (NAION)**.

## Overview
This tool bridges the gap between deep learning research and clinical utility. By utilizing a **U-Net architecture (ResNet34 backbone)**, the application segments the optic nerve head to extract critical metrics that identify the "Disk at Risk" phenotype.

### Key Features:
* **Automated Segmentation:** Precise masking of the Optic Disc and Cup.
* **Biomarker Extraction:** Real-time calculation of Superior Rim Thickness (px) and vCDR.
* **Vascular Profiling:** Automated Superior Vessel Density mapping.
* **Risk Stratification:** Logical flagging of "Crowded" vs. "Buffered" disc architectures.

## Methodology
* **Dataset:** 1,010 fundus images, including a diverse range of edematous and cupless phenotypes.
* **Model:** U-Net with a ResNet34 encoder, trained for 20 epochs to a validation accuracy of ~93.5%.
* **Deployment:** Python-based Streamlit dashboard optimized for high-resolution clinical photos.

## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `NAION_Risk_Unet_v1.pth` is in the root directory.
4. Run: `streamlit run naion_app.py`

## License
Distributed under the MIT License. See `LICENSE` for more information.

### Model Weights
The pre-trained weights for this project are hosted on **Hugging Face**:
[Download NAION_Risk_Unet_v1.pth](https://huggingface.co/datasets/jani2904/NAION-Risk-Analyzer/tree/main)

## Dashboard Previews

<img width="1855" height="1000" alt="Streamlit Webapp_Healthy eye" src="https://github.com/user-attachments/assets/be1e8523-8c38-4f09-9b02-51a1ab258d4f" />

<img width="1883" height="948" alt="Streamlitwebapp_disk_at_Risk_1" src="https://github.com/user-attachments/assets/1f6f56de-d45d-41d6-8b36-6ed267d6945e" />

<img width="1523" height="271" alt="Streamlitwebapp_disk_at_Risk_2" src="https://github.com/user-attachments/assets/9cef556d-7408-486f-aced-543791c94c1c" />





