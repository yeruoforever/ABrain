# ABrain

## 功能

### 模型库

* AlexNet
* VGG
* CAM
* Grad-CAM
* FCN
* UNet
* VNet
* SOTA

### 训练机

* GeneralTrainer（with or without deep supervised）
* CascadedTrainer

### 测量模块

* 体积测量
* 平面测量
* 误差测量

### 评估模块

* 模型质量评估
* 测量算法评估
* 一致性评估

### 数据集工具

* 数据集遍历分析
* 数据集Loader
* 数据集入选管理

### 数据增强及后处理Pipeline

* 重采样
* 裁剪、随机裁剪
* 缩放、随机缩放
* 模糊
* 拉伸、随机拉伸
* 旋转、随机旋转
* Maximum connected component
* 空洞填充
* 其它


## 代码约定

整体风格遵循[谷歌开源风格](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)规范, 默认使用`autopep8`作为代码格式化工具。

## 模块设置

## 单元测试

测试用例在`ABrain/tests/`下， 根据不同的功能写在不同的目录下，如：`ABrain/tests/models/factory.py`为模型工厂的测试用例。

测试框架为`unittest`, 测试指定模块使用如下命令：
```sh
python -m unittest <待测model>
```
如：
```bash
>>> python -m unittest ABrain/tests/models/factory.py
>>> ----------------------------------------------------------------------
>>> Ran 3 tests in 0.183s
>>> 
>>> OK
```

## 文档梳理
### 函数和类库注释
```markdown
一行简单的描述 <下跟一空行>

[多行详细描述，如果没必要可省略]

### Args：
- `参数名:<类型> [=默认值]`
    参数的细致介绍（有必要的话）。
- `in_channels:int`
    the number of input channels.
- `args`
    other positional parameters.
- `kwargs`
    other keywords parameters.
    
### Returns：
对返回值的描述。

### Example:
```python
>>> model = LeNet5(n_class=10, in_channels=1)
>>> x = torch.rand(8, 1, 32, 32)
>>> y = model(x)
>>> print(y.shape)
>>> torch.size([8, 10])
```

## 开发路线


