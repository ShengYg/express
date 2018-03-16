## 使用步骤
1、安装编译caffe，并链接到此目录下，然后编译Cython模块
```
ln -s caffe-fast-rcnn ${caffe_dir}
cd lib
make
cd ..
```

2、带有标注信息的txt文件放在`./data/express/`中

```
python experiments/scripts/help.py
```
在`./data/express/`生成`gt.pkl`

3、标注图片文件放在`./data/express/dataset/`中

```
python experiments/scripts/data_prepare.py
python experiments/scripts/phone_prepare.py
```
`data_prepare.py`处理定位信息，`phone_prepare.py`处理号码识别信息。

4、训练、测试定位网络
```
./experiments/scripts/train.sh
```
文件包含了训练和测试的代码，测试时的输入网络位置需要自己设定。

5、训练、测试识别网络
```
python tools/train_phone.py
python tools/test_phone.py
```
测试时的输入网络位置需要自己设定。

## 参数设置
所有默认参数在`lib/fast_fcnn/config.py`，重要的参数有英文注释。

当前网络使用的参数设置在`experiments/logs/*.yml`

`anchor`部分参数在`lib/rpn/generate_anchors.py`

参数设置参见论文[faster-rcnn](https://arxiv.org/abs/1506.01497)。

## 注意
1. 代码中包含诸多已注释的内容，是当初尝试的不同想法，可供参考。
1. 代码中有些部分使用的是绝对路径，使用时可能出错，根据实际情况修正即可。
