# 分析器
分析器用于逐`病例`分析影像的不同指标，可以用于统计像素、获取元信息和计算各种评估指标。

## Analyzer

### 方法和属性说明

|名称|类别|说明|
|---|---|---|
|name|属性|每个`Analyzer`都有一个名字，用于唯一标识当前分析器分析的内容。如`"Dice"`|
|analyze(subject)|方法|用于分析`病例`并返回结果，如计算$dice$值为`0.98`|
|parse_result(out)|方法|用于解析`analyze(subject)`返回的结果，并将结果拼接为字典的形式，如`{"Dice":0,98}`|

### 使用`Analyzer`类

当一个`Analyzer`以函数方式调用时，实际上是执行了这样的操作：
```python
analyzer = DiceAnalyzer()
analyzer.name                       # "Dice"

out = analyzer.analyze(subject)     # 0.98
parse_out = self.parse_result(out)  # {"Dice":0.98}
result = {"Subject": sid}           # {"Subject":"0001"}
result.update(parse_out)            # {"Subject":"0001", "Dice":0.98}
result                              # return
```
上述操作已经被抽象到`Analyzer`对象的`__call__`方法中，因此在实现`DiceAnalyzer`时只需实现`analyze()`方法即可实现如下操作。
```python
>>> analyzer = DiceAnalyzer()
>>> ans = analyzer(subject)
>>> print(ans)
>>> {"Subject":xxx, "Dice":0.98}
```
定义`DiceAnalyzer`:
```python
class DiceAnalyzer(Analyzer):
    def __init__(self):
        super().__init__(name="Dice")
    
    def analyze(self,subject:Subject):
        # 计算Dice
        dice = ......
        return dice
```

## ComposeAnalyzer

由多个`Analyzer`构成的分析器，一次执行多个分析。
内部实际等价于做了如下操作：
```python
ans = {}
for analyzer in self.analyzers:
    ans.update(analyzer(subject))
return ans 
```

### Example:
```python 
>>> analyzer = ComposeAnalyzer(
        SomeAnalyzer_1(),
        SomeAnalyzer_2(),
            ......
        SomeAnalyzer_n()
    )
>>> ans = analyzer(subject)
>>> print(ans)
>>> {
        "Subject": xxx,
        "Some_1" : xxx,
        "Some_2" : xxx,
            .......
        "Some_n" : xxx
    }

``` 
