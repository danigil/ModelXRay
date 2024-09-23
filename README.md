# ModelXRay
Python framework for AI model steganalysis research.  

Code repository for the research paper "Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning".  



Built on top of the [ZenML](https://github.com/zenml-io/zenml) framework.  

Features:
  - Creating and managing steganographic artifacts
    - Image representations from cover and stego data.
    - Currently supported cover data types:
      - [Pretrained keras CNNs](https://keras.io/api/applications/)
      - [MaleficNet](https://github.com/pagiux/maleficnet) cover models
      - [GHRP Model Zoos](https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations/tree/main)
  - Training FSL keras CNN feature extractors on created preprocessed image data
    - Binary classification (benign/malicious)
    - Two implemented testing methods:
      - 1NN
      - Centroid
    - Currently available CNN architectures:
      - [OSL CNN](https://paperswithcode.com/paper/siamese-neural-networks-for-one-shot-image)
      - [SRNet](https://doi.org/10.1109/TIFS.2018.2871749) CNN (Only works with 256x256 images)

# Licensing and Patent

Code in this repository is available for non-commercial use.  
This work is under US Provisional Patent Application No. 63/524,681.

Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

# Credits
Code in [External Code Dir](external_code/) is originally from other public GitHub repositories.  
  
MaleficNet: Code found in [MaleficNet Dir](external_code/maleficnet/maleficnet/) is code from [this](https://github.com/pagiux/maleficnet.git) repo with slight alterations.  

GHRP: Code found in [GHRP DIR](external_code/ghrp/) is code from [this](https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations.git) repo.
