# EMG_Fatigue_Monitoring
依赖: 

[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

[Keras](https://keras.io/) - CNN 分类可选

### 运行以训练和评估:
```
python FatigueMonitoring
```
编辑 main 函数下的代码以运行不同的评估场景

存储库中提供了用于本研究的已处理数据集

原始数据集可在以下位置获得: https://www.dropbox.com/s/daj4du1ra5zo821/Fatigue%20Data.zip?dl=0

### 数据格式

1.experiment_comparison文件夹包含格式为：C1：时间戳 C2：真实值 C3：预测值

2.Study1_medfilt11_EMG文件夹包含由窗口大小为 11 个样本的中值过滤器处理的 EMG 测量值

3.original_labels_Study1文件夹包含人类受试者提供的标签：0=不疲劳，1=疲劳

4.original_times_Study1文件夹包含每个脑电图测量的时间戳

5.对于代码中的问题，请查看注释

6.数据集文件格式：<user_id><exercise_id><repetition_id>

### 代码说明

#### 1. Classification主要用于分类器设计

##### `classifierWrapper(classifier, classifier_type, test_sample)`
- **功能**：根据指定的分类器类型对特征样本进行分类。
- **参数**：
  - `classifier`：分类器对象，支持多种类型，包括 `sklearn.svm.SVC`、kNN（在此库中定义）、`sklearn.ensemble.RandomForestClassifier`、`sklearn.ensemble.GradientBoostingClassifier` 或 `sklearn.ensemble.ExtraTreesClassifier`。
  - `classifier_type`：字符串，指定分类器的类型，可选 `"svm"`、`"knn"`、`"randomforest"`、`"gradientboosting"` 或 `"extratrees"`。
  - `test_sample`：要分类的特征向量（NumPy数组）。
- **输出**：返回一个列表 `[R, P]`，其中 `R` 代表分类结果的类别ID；`P` 代表分类结果的概率估计。

##### `normalizeFeatures(features)`
- **功能**：将一组特征矩阵进行标准化处理，使其具有0均值（mean）和1标准差（std），适用于大多数分类器训练场景。
- **参数**：
  - `features`：特征列表，每个元素是一个代表一组特征的特征矩阵（NumPy数组）。
- **返回值**：
  - `features_norm`：标准化后的特征矩阵列表。
  - `MEAN`：计算得到的特征均值向量。
  - `STD`：计算得到的特征标准差向量。
- **说明**：这个函数对特征数据进行标准化处理，有助于提高那些对特征缩放敏感的机器学习算法（如SVM和k-NN）的性能。

#### 2. DataCrossValidSplit

##### `split(datapath)`
- **功能**：将给定路径下的数据集分成训练集和测试集。创建多个数据折叠（fold），用于交叉验证。
- **参数**：
  - `datapath`：包含CSV文件的目录路径
- **说明**：test_ratio是测试集占总数据集的比例；Folds是交叉验证的折叠数。
##### `MidTermSplit(source, datapath, outFolderName)`
- **功能**：从指定路径读取CSV文件，对EMG进行中值滤波，提取特征，分类，保存npz文件
- **参数**：
  - `source`：数据来源标识，用于生成文件名和路径。
  - `datapath`：包含CSV文件的源数据路径
  - `outFolderName`：输出目录名，用于存储处理后的数据文件。
  
#### 3. FatigueMonitoring