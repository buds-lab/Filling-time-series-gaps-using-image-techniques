# Filling Time-Series Gaps Using Image Techniques: Multidimensional Context Autoencoder Approach for Building Energy Data Imputation

This repository contains all the necessary datasets and Jupyter notebooks used in our research, "Filling time-series gaps using image techniques: Multidimensional context autoencoder approach for building energy data imputation". 

## Research Abstract

In the era of Internet of Things (IoT), building energy prediction and management have seen significant progress. However, it often faces a substantial hurdle: the inconsistency and incompleteness of collected energy data from various sources. This issue can hinder accurate energy system predictions and management, and limit the effectiveness of the data for research and decision-making. 

Our research addresses this challenge by focusing on imputing the missing gaps in energy data. The uniqueness of this work lies in the application of state-of-the-art image-inpainting methods, such as Partial Convolution (PConv), in the field of energy data imputation. We exploit the regular patterns that energy data often exhibit to generate more accurate predictions for missing values. 

The research uses one of the largest publicly available whole building energy datasets, which includes 1479 power meters worldwide. Our findings suggest that advanced deep learning methods like Partial Convolution can significantly reduce the Mean Squared Error (MSE) in comparison with the traditional Convolutional neural networks (CNNs) and weekly persistence methods. This opens up a new perspective for employing time-series imaging in imputing energy data. The proposed imputation model is generalizable and scalable, making it an effective solution for filling missing energy data in both academic and industrial contexts.

## Repository Content

This repository is organized as follows:

1. **Notebooks**: This directory contains all Jupyter notebooks that provide the code implementations of the various techniques used in the research.

2. **Data**: Due to the constraints posed by the Github platform, the dataset used in our research is not provided here. Nevertheless, the energy dataset (including meter data, metadata, and weather data) could be found on [Kaggle website](https://www.kaggle.com/competitions/ashrae-energy-prediction/data) or [BDG2 Github repository](https://github.com/buds-lab/building-data-genome-project-2). Additionally, the labeled anomalies in meter readings can be found in the [repository of winning teams' solution](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/input/bad_meter_readings.zip).

## License

This project is licensed under the terms of the MIT license.

## Citation
If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{fu2024filling,
  title={Filling time-series gaps using image techniques: Multidimensional context autoencoder approach for building energy data imputation},
  author={Fu, Chun and Quintana, Matias and Nagy, Zoltan and Miller, Clayton},
  journal={Applied Thermal Engineering},
  volume={236},
  pages={121545},
  year={2024},
  publisher={Elsevier}
}
