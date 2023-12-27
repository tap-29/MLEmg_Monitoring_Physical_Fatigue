1.pywt的库实际上就是pip install PyWavelets
2.sklearn库实际上是pip install scikit-learn
3.classifier.py说明：
    RotationalForestClassifier:
        随机森林基础上添加了PCA
    initial_data用于将原始数据整形：
        X_train:样本数*特征数(30,30)
        Y_train:样本数*(30,)
    classifiers：
        分类器：输入对应的True，就会给相应的结果
    print_result:
        三个入参分别是：“打印表格的名字”，y_test,y_test_prdict
4.现在用get_data来调整数据