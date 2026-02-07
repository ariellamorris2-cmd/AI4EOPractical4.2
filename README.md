<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="centre">GEOL0069 AI for Earth Observation: Week 4 </h3>
<div align="center">
  <h3 align="center"><b>Altimetry Classification: Sea Ice vs. Leads</b></h3>

  <p align="center">
    Unsupervised Machine Learning for discriminating sea ice and leads using Sentinel-2 optical imagery and Sentinel-3 altimetry data.
    <br />
    <a href="https://github.com/Yariellamorris2-cmd/AI4EOPractical4.2"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/ariellamorris2-cmd/AI4EOPractical4.2/issues">Report Bug</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#Prerequisites">Prerequisites</a></li>
    <li><a href="#altimetry-classification">Altimetry Classification</a></li>
    <li><a href="#key-results">Key Results</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project explores the application of **Unsupervised Learning** to categorise satellite data. The core objective is to differentiate between **Sea Ice** and **Leads** (open water fractures in sea ice) using two distinct methods:
1. **Unit 1:** Structural setup and initial data exploration.
2. **Unit 2:** Implementation of K-Means and Gaussian Mixture Models (GMM) to classify Sentinel-3 altimetry echoes.

The Week 4 assignment for this module is to use these unsupervised learning methods for altimetry classification and, crucially, distinguishing between leads and sea ice in Sentinel-3 datasets. We will be focusing on Unit 2. This notebook has been annotated for your guidance. 

By analysing features such as **Peakiness (PP)**, **Stack Standard Deviation (SSD)**, and **Sigma_0**, we can effectively cluster radar waveforms without prior labelling, and then validate these clusters against official ESA classifications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
* [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
* [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![Rasterio](https://img.shields.io/badge/Rasterio-green?style=for-the-badge)](https://rasterio.readthedocs.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

This project utilises Google Colaboratory (GoogleColab), a cloud-based platform for writing, running, and sharing Python code. It provides free access to powerful TPU and GPU resources through your browser. Your personal Google Drive can easily integrate with Colab, making it very simple to store and share your work. Colab is extremely useful for data science, machine learning, and education. To access the notebook, click on the Google Colab link in the ipynb file included in this repository. 


### Prerequisites
* You will need a Python environment with the following libraries:

```sh
pip install netCDF4 numpy matplotlib scipy scikit-learn rasterio
```

* Mounting Google Drive on Google Colab 
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
  ```



## Altimetry Classification 

This is the process of identifying what kind of surface a satellite radar pulse has hit based on the "shape" of the returned signal (the waveform). When a satellite like Sentinel-3 or CryoSat-2 sends a radar pulse to Earth, the way that pulse bounces back tells us if it hit open water, lead, sea ice, or an ice sheet. We are processing net CDF files, and we are trying to distinguish between sea ice and leads. 


## Key Results

### 1. Average Echo Shapes
After clustering the Sentinel-3 waveforms using GMM, we generated the average echo shapes to visualise the physical differences between the classes.
* **Leads:** Characterised by very high peakiness and specular returns.
* **Sea Ice:** Characterised by more diffuse, wider waveforms.

![Average Echo Shapes](images/Mean_Echo_Shapes.png)

> **Insight:** The shaded areas in the plot represent the standard deviation of the waveforms, illustrating the variability within each surface class.

### 2. Validation (Confusion Matrix)
To quantify the performance of the Unsupervised Learning model, we compared our clusters against the ESA official surface type classification.


Below is a confusion matrix comparing the ESA official classification (flags) against my GMM cluster classification:

![Confusion Matrix](images/Confusion_Matrix.png)


> **Insight:** The Gaussian Mixture Model (GMM) performed exceptionally well, distinguishing leads from sea ice with 99.62% accuracy. Most discrepancies were negligible, with only 22 instances of sea ice misclassified as leads and 24 instances of leads misclassified as sea ice. This high level of agreement with the ESA official classification validates the use of Peakiness, Sigma_0, and SSD as robust features for unsupervised altimetry classification.


### Acknowledgements

**European Space Agency (ESA):** For providing the Sentinel-2 optical imagery and Sentinel-3 SRAL altimetry datasets.

**Alfred Wegener Institute (AWI):** For the methodologies regarding physical waveform alignment.

**GEOL0069 Module Team:** For providing the framework for AI applications in Earth Observation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

