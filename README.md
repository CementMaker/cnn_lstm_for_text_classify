 # <center>DNN for text classify</center>

## <center> 中文文本分类 <\center>

1. 文本采用压缩包里面的文档，运行代码之前需要解压文件夹
   解压之后的文件夹名称:corpus

2. CNN_SRC
   1. PreProcess.py 数据预处理，做神经网络的输入
   2. cnn.py textCnn模型
   3. get_text_meaning.py 删除编码错误的文件，这个非常粗暴
   4. train.py 训练数据，看网络的效果
  
3. rnn_src
   1. PreProcess.py 数据预处理，做神经网络的输入
   2. cnn.py lstm模型
   3. train.py 训练数据，看网络的效果


执行过程：
先执行PreProcess.py
再执行train.py