# Real3D-AD: A Dataset of Point Cloud Anomaly Detection

Jiaqi Liu*, Guoyang Xie*, Ruitao Chen*, Xinpeng Li, Jinbao Wang†, Yong Liu, Chengjie Wang, and Feng Zheng†

(* Equal contribution; † Corresponding authors)

Our paper is summitted to NeurIPS 2023 Datasets & Benchmarks Track. [[Paper]]()

# Overview
This project aims to construct a new dataset of **high-resolution 3D point clouds** for anomaly detection tasks in real-world scenes.

**Real3D-AD** can be used for training and testing 3D anonmaly detection algorithms.

Note that different from RGB + Depth patterns, we **only** provide 3D point clouds for users.


# Real3D-AD

<img src="./doc/real3d.png" width=900 alt="Real3D Dataset" align=center>


## Summary
+ overview of all classes in Real3D-AD

Real3D-AD comprises a total of 1,254 samples that are distributed across 12 distinct categories. These categories include Airplane, Car, Candybar, Chicken, Diamond, Duck, Fish, Gemstone, Seahorse, Shell, Starfish, and Toffees.


## Download

+ To download the Real3D-AD dataset (Dataset for training and evaluation, pcd format), click [real3d-ad-pcd.zip(google drive)](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing) or [real3d-ad-pcd.zip(baidu disk: vrmi)](https://pan.baidu.com/s/1orQY3DjR6Z0wazMNPysShQ)
+ To download the Real3D-AD dataset (Source data from camera, ply format), click [real3d-ad-ply.zip(google drive)](https://drive.google.com/file/d/1lHjvyVquuO8-ROOYcnf7O_lliL1Wa36V/view?usp=sharing) or [real3d-ad-ply.zip(baidu disk：vvz1)](https://pan.baidu.com/s/1BRdJ8oSwrpAPxTOEwUrjdw)


### Data preparation
- Download real3d-ad-pcd.zip and extract into `./data/`
```
data
├── airplane
    ├── train
        ├── 1_prototype.pcd
        ├── 2_prototype.pcd
        ...
    ├── test
        ├── 1_bulge.pcd
        ├── 2_sink.pcd
        ...
    ├── gt
        ├── 1_bulge.txt
        ├── 2_sink.txt
        ... 
├── car
...
```

### Checkpoint preparation

| Backbone          | Pretrain Method                                                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Point Transformer | [Point-MAE](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth)                                                                                       |
| Point Transformer | [Point-Bert](https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1)                                                                                                        |

- Download checkpoints and move them into `./checkpoints/`


## Dataset Statistic

+ brief describe our dataset in Table
  
| Source | Class         | Real Size [mm] (length/width/height) | Transparency | TrainingNum (good) | TestNum (good) | TestNum (defect) | TotalNum | TrainingPoints (min/max/mean) | TestPoints (min/max/mean) | AnomalyProportion Δ |
|:--------:|---------------|----------------------------:|--------------:|-----------------:|-------------:|---------------:|-------:|----------------------------:|---------------------------:|--------:|
| 1      | Airplane      |     34.0/14.2/31.7         |      Yes     |               4 |          50 |            50 |   104 |       383k/ 413k/ 400k     |       168k/ 773k/351k     |  1.17% |
| 2      | Car           |     35.0/29.0/12.5         |      Yes     |               4 |          50 |            50 |   104 |       566k/1296k/1097k     |        90k/ 149k/131k     |  1.98% |
| 3      | Candybar      |     33.0/20.0/ 8.0         |      Yes     |               4 |          50 |            50 |   104 |       339k/1183k/ 553k     |       149k/ 180k/157k     |  2.36% |
| 4      | Chicken       |     25.0/14.0/20.0         | No (white)   |               4 |          52 |            54 |   110 |       217k/1631k/1157k     |        87k/1645k/356k     |  4.46% |
| 5      | Diamond       |     29.0/29.0/18.7         |      Yes     |               4 |          50 |            50 |   104 |      1477k/2146k/1972k     |        66k/  84k/ 75k     |  5.40% |
| 6      | Duck          |     30.0/22.2/29.4         |      Yes     |               4 |          50 |            50 |   104 |       545k/2675k/1750k     |       155k/ 784k/216k     |  1.99% |
| 7      | Fish          |     37.7/24.0/ 4.0         |      Yes     |               4 |          50 |            50 |   104 |       230k/ 251k/ 240k     |       104k/ 117k/110k     |  2.85% |
| 8      | Gemstone      |     22.5/18.8/17.0         |      Yes     |               4 |          50 |            50 |   104 |       169k/1819k/ 835k     |        43k/ 645k/104k     |  2.06% |
| 9      | Seahorse      |     38.0/11.2/ 3.5         |      Yes     |               4 |          50 |            50 |   104 |       189k/ 203k/ 194k     |        74k/  90k/ 83k     |  4.57% |
| 10     | Shell         |     21.7/22.0/ 7.7         |      Yes     |               4 |          52 |            48 |   104 |       280k/ 316k/ 295k     |       110k/ 144k/125k     |  2.25% |
| 11     | Starfish      |     27.4/27.4/ 4.8         |      Yes     |               4 |          50 |            50 |   104 |       198k/ 209k/ 202k     |        74k/ 116k/ 88k     |  4.46% |
| 12     | Toffees       |     38.0/12.0/10.0         |      Yes     |               4 |          50 |            50 |   104 |       178k/1001k/ 385k     |        78k/  97k/ 88k     |  2.46% |

(Δ: Mean proportion of abnormal point clouds in Test set)

## Data Collection

+ description of instruments

<img src="./doc/instruments.png" width=300 alt="instruments" align=center>
The PMAX-S130 optical system comprises a pair of lenses with low distortion properties, a high luminance LED, and a blue-ray filter. The blue light scanner is equipped with a lens filter that selectively allows only the blue light of a specific wavelength to pass through. The filter effectively screens the majority of blue light due to its relatively low concentration in both natural and artificial lighting. Nevertheless, using blue light-emitting light sources could pose a unique obstacle in this context. The image sensor can collect light using the lens aperture. Hence, the influence exerted by ambient light is vastly reduced.

+ how to capture point clouds and complete one prototype

<img src="./doc/make_prototypes.png" width=900 alt="make prototype" align=center>
Initially, the stationary object undergoes scanning while the turntable completes a full revolution of 360°, enabling the scanner to capture images of the various facets of the object. Subsequently, the object undergoes reversal, and the process of rotation and scanning is reiterated. Following the manual calibration of the front and back scanning outcomes, the algorithm performs a precise calibration of the stitching process. If there are any gaps in the stitching outcome, the scan stitching process is reiterated until the point cloud is rendered.

+ anomalies

The anomalies pertaining to point clouds can be classified into two categories: incompleteness and redundancy. In the dataset, we named them bulge and sink. Besides, more samples are made by copying and cutting edges.


## Annotation
+ how to annotate

The collected point clouds are annotated using CloudCompare software
CloudCompare is a 3D point cloud (grid) editing and processing software. Originally, it was designed to directly compare dense three-dimensional point clouds. It relies on a specific octree structure and provides excellent performance for tasks such as point cloud comparison.
The anotation process of point cloud is shown in the figure below.
<!-- ![image-20230605141032952](https://github.com/M-3LAB/H3D-AD/blob/main/doc/anotation.png) -->

<img src="./doc/annotation.png" width=900 alt="Anotation phase" align=center>

## Benchmark

+ beseline methods
  
We take BTF and M3DM as basic baseline methods, and improve baseline using PatchCore and .

+ metrics
  
We choose AUROC and AUPU as metric for object level and point level anomaly detection.

+ benchmark results

| Object AUROC | BTF_FPFH | BTF_Raw | M3DM_PointMAE | M3DM_PointBERT | PatchCore+FPFH | PatchCore+FPFH+raw | PatchCore+PointMAE | Our baseline |
|--------------|---------|----------|---------------|----------------|----------------|--------------------|--------------------|--------------|
| airplane     | 0.730   | 0.520    | 0.434         | 0.407          | 0.882          | 0.848              | 0.726              | 0.716        |
| car          | 0.647   | 0.560    | 0.541         | 0.506          | 0.590          | 0.777              | 0.498              | 0.697        |
| candybar     | 0.703   | 0.462    | 0.450         | 0.442          | 0.565          | 0.626              | 0.585              | 0.827        |
| chicken      | 0.789   | 0.432    | 0.683         | 0.673          | 0.837          | 0.853              | 0.827              | 0.852        |
| diamond      | 0.707   | 0.545    | 0.602         | 0.627          | 0.574          | 0.784              | 0.783              | 0.900        |
| duck         | 0.691   | 0.784    | 0.433         | 0.466          | 0.546          | 0.628              | 0.489              | 0.584        |
| fish         | 0.602   | 0.549    | 0.540         | 0.556          | 0.675          | 0.837              | 0.630              | 0.915        |
| gemstone     | 0.686   | 0.648    | 0.644         | 0.617          | 0.370          | 0.359              | 0.374              | 0.417        |
| seahorse     | 0.596   | 0.779    | 0.495         | 0.494          | 0.505          | 0.767              | 0.539              | 0.762        |
| shell        | 0.396   | 0.754    | 0.694         | 0.577          | 0.589          | 0.663              | 0.501              | 0.583        |
| starfish     | 0.530   | 0.575    | 0.551         | 0.528          | 0.441          | 0.471              | 0.519              | 0.506        |
| toffees      | 0.539   | 0.630    | 0.552         | 0.562          | 0.541          | 0.570              | 0.663              | 0.685        |
| Average      | 0.643   | 0.601    | 0.552         | 0.538          | 0.597          | 0.692              | 0.588              | 0.705        |

| Object AUPR | BTF_FPFH | BTF_Raw | M3DM_PointMAE | M3DM_PointBERT | PatchCore+FPFH | PatchCore+FPFH+raw | PatchCore+PointMAE | Our baseline |
|-------------|---------|----------|---------------|----------------|----------------|--------------------|--------------------|--------------|
| airplane    | 0.659   | 0.506    | 0.479         | 0.497          | 0.852          | 0.807              | 0.747              | 0.703        |
| car         | 0.653   | 0.523    | 0.508         | 0.517          | 0.611          | 0.766              | 0.555              | 0.753        |
| candybar    | 0.638   | 0.490    | 0.498         | 0.480          | 0.553          | 0.611              | 0.576              | 0.824        |
| chicken     | 0.814   | 0.464    | 0.739         | 0.716          | 0.872          | 0.885              | 0.864              | 0.884        |
| diamond     | 0.677   | 0.535    | 0.620         | 0.661          | 0.569          | 0.767              | 0.801              | 0.884        |
| duck        | 0.620   | 0.760    | 0.533         | 0.569          | 0.506          | 0.560              | 0.488              | 0.588        |
| fish        | 0.638   | 0.633    | 0.525         | 0.628          | 0.642          | 0.844              | 0.720              | 0.939        |
| gemstone    | 0.603   | 0.598    | 0.663         | 0.628          | 0.411          | 0.411              | 0.444              | 0.454        |
| seahorse    | 0.567   | 0.793    | 0.518         | 0.491          | 0.508          | 0.763              | 0.546              | 0.787        |
| shell       | 0.434   | 0.751    | 0.616         | 0.638          | 0.573          | 0.553              | 0.590              | 0.646        |
| starfish    | 0.557   | 0.579    | 0.573         | 0.573          | 0.491          | 0.473              | 0.561              | 0.491        |
| toffees     | 0.505   | 0.700    | 0.593         | 0.569          | 0.506          | 0.559              | 0.708              | 0.721        |
| Average     | 0.624   | 0.603    | 0.572         | 0.581          | 0.599          | 0.676              | 0.626              | 0.723        |

| Point AUROC | BTF_FPFH | BTF_Raw | M3DM_PointMAE | M3DM_PointBERT | PatchCore+FPFH | PatchCore+FPFH+raw | PatchCore+PointMAE | Our baseline |
|-------------|---------|----------|---------------|----------------|----------------|--------------------|--------------------|--------------|
| airplane    | 0.738   | 0.564    | 0.530         | 0.523          | 0.471          | 0.556              | 0.579              | 0.631        |
| car         | 0.708   | 0.647    | 0.607         | 0.593          | 0.643          | 0.740              | 0.610              | 0.718        |
| candybar    | 0.864   | 0.735    | 0.683         | 0.682          | 0.637          | 0.749              | 0.635              | 0.724        |
| chicken     | 0.693   | 0.608    | 0.735         | 0.790          | 0.618          | 0.558              | 0.683              | 0.676        |
| diamond     | 0.882   | 0.563    | 0.618         | 0.594          | 0.760          | 0.854              | 0.776              | 0.835        |
| duck        | 0.875   | 0.601    | 0.678         | 0.668          | 0.430          | 0.658              | 0.439              | 0.503        |
| fish        | 0.709   | 0.514    | 0.600         | 0.589          | 0.464          | 0.781              | 0.714              | 0.826        |
| gemstone    | 0.891   | 0.597    | 0.654         | 0.646          | 0.830          | 0.539              | 0.514              | 0.545        |
| seahorse    | 0.512   | 0.520    | 0.561         | 0.574          | 0.544          | 0.808              | 0.660              | 0.817        |
| shell       | 0.571   | 0.489    | 0.748         | 0.732          | 0.596          | 0.753              | 0.725              | 0.811        |
| starfish    | 0.501   | 0.392    | 0.555         | 0.563          | 0.522          | 0.613              | 0.641              | 0.617        |
| toffees     | 0.815   | 0.623    | 0.679         | 0.677          | 0.411          | 0.549              | 0.727              | 0.759        |
| Average     | 0.722   | 0.566    | 0.637         | 0.636          | 0.592          | 0.692              | 0.634              | 0.700        |

| Point AUPR | BTF_FPFH | BTF_Raw | M3DM_PointMAE | M3DM_PointBERT | PatchCore+FPFH | PatchCore+FPFH+raw | PatchCore+PointMAE | Our baseline |
|------------|---------|----------|---------------|----------------|----------------|--------------------|--------------------|--------------|
| airplane   | 0.027   | 0.012    | 0.007         | 0.007          | 0.027          | 0.016              | 0.016              | 0.017        |
| car        | 0.028   | 0.014    | 0.018         | 0.017          | 0.034          | 0.160              | 0.069              | 0.135        |
| candybar   | 0.118   | 0.025    | 0.016         | 0.016          | 0.142          | 0.092              | 0.020              | 0.109        |
| chicken    | 0.044   | 0.049    | 0.310         | 0.377          | 0.040          | 0.045              | 0.052              | 0.044        |
| diamond    | 0.239   | 0.032    | 0.033         | 0.038          | 0.273          | 0.363              | 0.107              | 0.191        |
| duck       | 0.068   | 0.020    | 0.011         | 0.011          | 0.055          | 0.034              | 0.008              | 0.010        |
| fish       | 0.036   | 0.017    | 0.025         | 0.039          | 0.052          | 0.266              | 0.201              | 0.437        |
| gemstone   | 0.075   | 0.014    | 0.018         | 0.017          | 0.093          | 0.066              | 0.008              | 0.016        |
| seahorse   | 0.027   | 0.031    | 0.030         | 0.028          | 0.031          | 0.291              | 0.071              | 0.182        |
| shell      | 0.018   | 0.011    | 0.022         | 0.021          | 0.031          | 0.049              | 0.043              | 0.065        |
| starfish   | 0.034   | 0.017    | 0.040         | 0.040          | 0.037          | 0.035              | 0.046              | 0.039        |
| toffees    | 0.055   | 0.016    | 0.021         | 0.018          | 0.040          | 0.055              | 0.055              | 0.067        |
| Average    | 0.065   | 0.022    | 0.046         | 0.052          | 0.074          | 0.129              | 0.058              | 0.113        |

## How to reproduce our benchmark

+ environment preparation

We implement benchmark under CUDA 11.3
Our environment can be reproduced by the following command:

```
conda env create -f real3dad.yaml
# Note that point2_ops_lib may need to be installed by the following command：
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

+ how to train and evaluation

```
sh start.sh
```
The result will output four different metrics (Object/Point AUROC/AUPR).

+ how to visualize abnormal map

In util.visualization.py, we provide the function ''vis_pointcloud_gt'' to visualize ground truth with our gt files. Also, with a saved anomaly map xx.npy(n values between 0 and 1) and the corresponding pcd file(n xyz points), you can use "vis_pointcloud_anomalymap(point_cloud, anomaly_map)" to visualize anomaly regions.

## Thanks

Our benchmark is built on [BTF](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM) and [PatchCore](https://github.com/amazon-science/patchcore-inspection), thanks their extraordinary works!


## License
The dataset is released under the CC BY 4.0 license.
