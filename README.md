<a id="readme-top"></a>

<br />
<div align="center">
  <h1 align="centre">GEOL0069 AI for Earth Observation: Week 4 </h3>
<div align="center">
  <h2 align="center"><b>Altimetry Classification: Sea Ice vs. Leads</b></h3>

  <p align="center">
    Unsupervised Machine Learning for discriminating sea ice and leads using Sentinel-2 optical imagery and Sentinel-3 altimetry data.
    <br />
    <a href="https://github.com/ariellamorris2-cmd/AI4EOPractical4.2"><strong>Explore the docs »</strong></a>
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
1. **Unit 2:** Implementation of K-Means and Gaussian Mixture Models (GMM) to classify Sentinel-3 altimetry echoes.

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

## Code Explanation 

The technical core of this project involves transforming raw satellite waveforms into actionable actionable insights through statistical clustering and rigorous validation.

### 1. Statistical Waveform Analysis
Once the Guassian Mixture Model (GMM) has partitioned the data into two clusters, we must verify that these clusters correspond to the physical reality of "Sea Ice" and "Leads."

  ```sh
# Calculating bin-wise mean and standard deviation for each cluster
mean_ice = np.mean(waves_cleaned[clusters_gmm==0], axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)
 ```

Boolean Indexing: We use clusters_gmm==0, to create a mask, isolating only the waveforms the model identified as Class 0. 
Axis-wise Statistics: By calculating the mean on axis=0, we find the average power for each of the 256 "bins" across all signals that cluster
The Physical Link: 
- Leads (Class 1)
- Sea Ice (Class 0)
- Variability Visualisation: the plt.fill_between function creates the shaded area representing the standard deviation. A narrow shaded area indicates that the echoes within that cluster are highly consistent in shape.

### 2. Validation with Confusion Matrix 
To prove the model's reliability, we compare the "unsupervised" GMM predictions against the ESA Official Surface Type Classification.
  ```sh
# Generating a Confusion Matrix to compare GMM results vs. ESA Flags
conf_matrix = confusion_matrix(true_labels, predicted_gmm)
print(conf_matrix)
 ```

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

## Conclusion
The implementation of the **Gaussian Mixture Model (GMM)** proved highly effective for the discrimination of sea ice and leads using Sentinel-3 altimetry data. By utilizing a feature space consisting of **Peakiness**, **Sigma_0**, and **Stack Standard Deviation (SSD)**, the model achieved a validation accuracy of **99.62%** when compared against official ESA surface type classifications.

### Key Findings:

* **Waveform Physics:** The high accuracy confirms that the selected features are robust proxies for surface roughness. Leads produce a distinct specular signature with low variability, whereas sea ice produces diffuse returns with a higher standard deviation across the waveform.
* **Model Suitability:** GMM’s ability to model cluster covariance provided a superior fit for the correlated nature of radar backscatter features compared to simpler methods like K-Means.
* **Alignment Impact:** The physical alignment of waveforms (AWI-style) significantly reduced the standard deviation in peak positions (from **10.77 to 8.19 bins**), leading to much cleaner average echo shapes and more reliable cluster centroids.

This project demonstrates that unsupervised machine learning can effectively automate the classification of large-scale satellite datasets, providing a reliable alternative to manual labelling for polar region monitoring.

### Acknowledgements

**European Space Agency (ESA):** For providing the Sentinel-2 optical imagery and Sentinel-3 SRAL altimetry datasets.

**Alfred Wegener Institute (AWI):** For the methodologies regarding physical waveform alignment.

**GEOL0069 Module Team:** For providing the framework for AI applications in Earth Observation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

