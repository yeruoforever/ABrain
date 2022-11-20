# STL2NII

将`*.STL`文件记录的3D图像标注转换为`*.nii.gz`。

- `*.STL`文件记录了一个3D体的表面信息，记录了构成这个表面的所有顶点和三角面。
- `*。nii.gz`文件记录了构成3D体的体素点阵。

格式转换的原理是：

已知一个3D体的闭合表面`Mesh`， 找出当前空间坐标系中所有在表面内部的点，并记录为`Volume`。

## 软件依赖

|程序|描述|URL|
|--|--|:--|
|Slicer3D|提供文件读写及格式转换功能。|[Main Page](https://www.slicer.org/)|

## 文件夹结构

- 原始图像为`xxx.nii.gz`,其中记录了影像空间信息，用于构建空间坐标系和网格采样。
- 标注文件为`xxx.stl`,为`Mesh`形式的待转换图像分割标注文件。
- 多个文件对`(xxx.nii.gz, xx.stl)`存放在文件夹`<input_dir>`中。
- 转换完成的文件为同名文件`xxx.nii.gz`，存放在文件夹`<output_dir>`中。

```sh
>>>
    <input_dir>
        |- xxx.stl
        |- xxx.nii.gz
        |- yyy.stl
        |- yyy.nii.gz
        |- ......
>>>
    <output_dir>
        |- xxx.nii.gz
        |_ yyy.nii.gz
        |_ ......
```
## 使用方法

<video id="video" width="1024" height="768" controls>
    <source id="howtouse" src="./stl2nii.mov" type="video/mp4">
</video>