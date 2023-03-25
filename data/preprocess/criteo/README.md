```
.
├── criteo_statis_info.pickle // 特征处理需要的中间数据，非必须，因为调参时需要多次运行代码，存储起来直接读的话就不需要重新读取大规模数据集然后计算。
├── criteo_test.txt           // 经过特征处理后的测试集
├── criteo_train.txt          // 经过特征处理后的训练集
├── criteo_train_valid.txt    // 经过特征处理后的训练集+验证集
├── criteo_valid.txt          // 经过特征处理后验证集
└── README.md                 // 注释文件

0 directories, 6 files
```