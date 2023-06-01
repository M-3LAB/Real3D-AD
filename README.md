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
  
| Source | Class         | Real Size (length/width/height) | Transparency | TrainingNum (good) | TestNum (good) | TestNum (defect) | TotalNum | TrainingPoints (min/max/mean) | TestPoints (min/max/mean) | Anomaly Proportion Δ |
|--------|---------------|----------------------------|--------------|-----------------|-------------|---------------|-------|----------------------------|---------------------------|--------|
| 1      | airplane      |                            |              |               4 |          50 |            50 |   104 |       383k /  413k /  400k |       168k /  773k / 351k | 1.17% |
| 2      | barCandy      |                            |              |               4 |          50 |            50 |   104 |       178k / 1001k /  385k |        78k /   97k /  88k | 2.46% |
| 3      | butterflyFish |                            |              |               4 |          50 |            50 |   104 |       230k /  251k /  240k |       104k /  117k / 110k | 2.85% |
| 4      | chicken       |                            |              |               4 |          26 |            27 |    57 |       217k / 1631k / 1157k |        87k / 1645k / 356k | 4.46% |
| 5      | duck          |                            |              |               4 |          50 |            50 |   104 |       545k / 2675k / 1750k |       155k /  784k / 216k | 1.99% |
| 6      | gemstone      |                            |              |               4 |          50 |            50 |   104 |       169k / 1819k /  835k |        43k /  645k / 104k | 2.06% |
| 7      | pacCandy      |                            |              |               4 |          50 |            50 |   104 |       339k / 1183k /  553k |       149k /  180k / 157k | 2.36% |
| 8      | seaHorse      |                            |              |               4 |          50 |            50 |   104 |       189k /  203k /  194k |        74k /   90k /  83k | 4.57% |
| 9      | shell         |                            |              |               4 |          52 |            48 |   104 |       280k /  316k /  295k |       110k /  144k / 125k | 2.25% |
| 10     | smallDiamond  |                            |              |               4 |          50 |            50 |   104 |      1477k / 2146k / 1972k |        66k /   84k /  75k | 5.40% |
| 11     | sportsCar     |                            |              |               4 |          50 |            50 |   104 |       566k / 1296k / 1097k |        90k /  149k / 131k | 1.98% |
| 12     | starFish      |                            |              |               4 |          50 |            50 |   104 |       198k /  209k /  202k |        74k /  116k /  88k | 4.46% |

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


## Annotation
+ how to annotate



## Benchmark

+ task definition
+ structure of dataset
+ beseline method


## Training and Evaluation

+ how to train


+ how to evaluate