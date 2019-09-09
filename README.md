 # <center>DNN for text classify</center>

## <center> 中文文本分类 </center>

<pre><code>
├── Model                 模型文件夹
│   ├── FastText.py       fastText算法，调用好接口
│   ├── NeuralBOW.py      NBOW算法，可以作为base
│   ├── __init__.py
│   ├── textCnn.py        textCNN
│   └── textLstm.py       textRnn，这里使用lstm
├── README.md
├── __init__.py
├── data
│   ├── fastTextData      fastText对应的数据
│   │   ├── train_data    训练集
│   │   └── valid_data    测试集
│   └── pkl               DNN对应的处理好的数据
│       ├── test.pkl      DNN对应的测试集
│       └── train.pkl     DNN对应的训练集
├── data.tar              源数据
├── preprocess.py         预处理数据，将源文本数据处理成NN和fastText所需要的特征
└── train.py              训练模型，将每个NN模型对应训练测试过程封装成类
'''

文本采用压缩包里面的文档，运行代码之前需要解压文件夹
解压之后的文件夹名称: data/context


执行过程：
先执行PreProcess.py
再执行train.py

代码中的参数为不走心设置，可以自行进行调参
    包括epoches，mini batch大小，学习率和每个模型对应的模型参数等

后续工作：
    加入多层LSTM，dynamic_rnn，attention机制

加入项目：
   neural machine translate
</code></pre>
