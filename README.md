# 口腔鳞状细胞癌病理分期系统

# 简介

<img src="docs\imgs\oscc.png" alt="oscc" style="zoom:80%;" />

口腔鳞状细胞癌 (Oral Squamous Cell Carcinoma) 是头颈部最常见的恶性肿瘤，占所有口腔恶性肿瘤的90%以上，其发病起源于口腔黏膜表皮区域，严重情况下将影响患者正常生活，出现无法说话、进食等情况。

## 病理分期标准

<img src="docs\imgs\stage.png" alt="stage" style="zoom:50%;" />

## 浸润深度测量

医生通过镜头观测黏膜、肿瘤的位置，选择离肿瘤最近的正常黏膜的基底膜的位置，用水笔点出该位置（0.4-0.5mm）（差不多半个到整个黏膜的厚度），拉出基准线，然后确定肿瘤最深处到基准线的垂直距离，作为浸润深度。

## OSCC数据集



# 系统指南

## 主页

![base](docs\imgs\base.png)

## 数据导入

![load](docs\imgs\load.png)

## 数据处理

![process](docs\imgs\process.png)

## 可视化

![visualize](docs\imgs\visualize.png)

# 代码

## Requirements

* django
* pytorch
* openslide
* cv2
* skimage
* albumentations

## Usage

### 普通访问

1 `python manage.py runserver`

2 访问对应端口：v0 / admin

### 添加超级用户

1  `python manage.py createsuperuser`

2 输入用户名和密码

3 进入数据管理后台页面

## 代码分析

<img src="docs\imgs\code.png" alt="code" style="zoom: 33%;" />