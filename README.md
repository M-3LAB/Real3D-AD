# H3D-AD: A Dataset of High-Resolution 3D Anomaly Detection Tasks

Jiaqi Liu*, Guoyang Xie*, Jinbao Wang†, Ruitao Chen, Xinpeng Li, Yong Liu, Chengjie Wang, and Feng Zheng†

(* Equal contribution; † Corresponding authors)

Our paper is summitted to NeurIPS 2023 Dataset & Benchmark Track. [[Paper]]()

# Overview
This project aims to construct a new dataset of **high-resolution 3D point clouds** for anomaly detection tasks in real-world scenes.

**H3D-AD** can be used for training and testing 3D anonmaly detection algorithms.

Note that different from RGB + Depth patterns, we **only** provide 3D point clouds for users.


# H3D-AD

## Summary
+ draw object overview of all classes in H3AD-AD (give color snapshot and 3d point clouds)

+ summary


## Download

+ To download the H3D-AD dataset (Source, pcb format), click [h3d-ad.pcb.zip]()
+ To download the H3D-AD dataset (Source, ply format), click [h3d-ad.ply.zip]()
+ To download the H3D-AD dataset (Down-sampled, pcb format), click [h3d-ad-ds.pcb.zip]()

## Dataset Statistic

+ brief describe our dataset in Table
  
| Source | Class         | Size (length/width/height) | Transparency | TrainingNum (good) | TestNum (good) | TestNum (defect) | TotalNum | TrainingPoints (min/max/mean) | TestPoints (min/max/mean) | Anomaly Proportion Δ |
|--------|---------------|----------------------------|--------------|-----------------|-------------|---------------|-------|----------------------------|---------------------------|--------|
| 1      | airplane      |                            |              |                 |             |               |       |                            |                           |        |
| 2      | barCandy      |                            |              |                 |             |               |       |                            |                           |        |
| 3      | butterflyFish |                            |              |                 |             |               |       |                            |                           |        |
| 4      | chicken       |                            |              |                 |             |               |       |                            |                           |        |
| 5      | duck          |                            |              |                 |             |               |       |                            |                           |        |
| 6      | gemstone      |                            |              |                 |             |               |       |                            |                           |        |
| 7      | pacCandy      |                            |              |                 |             |               |       |                            |                           |        |
| 8      | seaHorse      |                            |              |                 |             |               |       |                            |                           |        |
| 9      | shell         |                            |              |                 |             |               |       |                            |                           |        |
| 10     | smallDiamond  |                            |              |                 |             |               |       |                            |                           |        |
| 11     | sportsCar     |                            |              |                 |             |               |       |                            |                           |        |
| 12     | starFish      |                            |              |                 |             |               |       |                            |                           |        |

(Δ: Mean proportion of abnormal point clouds in Test set)

+ range of data coordinates
+ objects attributes (shape, size, transparency, etc.)
+ abnormal types
+ 

## Data Collection

+ description of instruments
+ how to capture point clouds and complete one tamplate
+ how to make anomalies
+ labor and time consuming


## Annotation and format
+ how to annotate



## Benchmark

+ task definition
+ structure of dataset
+ beseline method


## Training and Evaluation

+ how to train


+ how to evaluate