

```python
import numpy as np
a=np.arange(5)
a.dtype
```




    dtype('int32')




```python
a
```




    array([0, 1, 2, 3, 4])




```python
a.shape#该向量有5个元素，它们的值分别是从0到4。该数组的shape属性是一个元组，存放的是数组在每一个维度的长度。
```




    (5,)




```python
#创建多维数组
m=np.array([np.arange(2),np.arange(2)])
m
```




    array([[0, 1],
           [0, 1]])




```python
m.shape#2x2的数组
```




    (2, 2)




```python
#选择numpy数组元素
b=np.array([[1,2],[3,4]])
b
```




    array([[1, 2],
           [3, 4]])




```python
b[0,0]
```




    1




```python
b[0,1]
```




    2




```python
b[1,0]
```




    3




```python
b[1,1]
```




    4




```python
#字符码，浮点型
np.arange(7,dtype='f')
```




    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.], dtype=float32)




```python
#复数型
np.arange(7,dtype='D')
```




    array([ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  5.+0.j,  6.+0.j])




```python
#dtype属性
t=np.dtype('Float64')
t.char
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Numeric-style type codes are deprecated and will result in an error in the future.
      
    




    'd'




```python
t.type
```




    numpy.float64




```python
t.str#f8代表float64
```




    '<f8'




```python
#一维数组的切片与索引
c=np.arange(9)
c
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8])




```python
c[3:7]
```




    array([3, 4, 5, 6])




```python
c[:7:2]#顾首不顾尾
```




    array([0, 2, 4, 6])




```python
c[::-1]#反转数组
```




    array([8, 7, 6, 5, 4, 3, 2, 1, 0])




```python
#处理数组形状
d=np.arange(24).reshape(2,3,4)
d
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
d.ravel()#拆解，将多维变为一维
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23])




```python
d.flatten()#拉直，功能与ravel（）相同
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23])




```python
d.shape=(6,4)#用元组指定数组形状
d
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])




```python
d.transpose()#转置
```




    array([[ 0,  4,  8, 12, 16, 20],
           [ 1,  5,  9, 13, 17, 21],
           [ 2,  6, 10, 14, 18, 22],
           [ 3,  7, 11, 15, 19, 23]])




```python
d.resize((2,12))#调整大小
d
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])




```python
#堆叠数组
e=np.arange(9).reshape(3,3)
e
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
f=2*e
f
```




    array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])




```python
#水平叠加
np.hstack((e,f))
```




    array([[ 0,  1,  2,  0,  2,  4],
           [ 3,  4,  5,  6,  8, 10],
           [ 6,  7,  8, 12, 14, 16]])




```python
#用concatenate()函数也能达到同样的效果
np.concatenate((e,f),axis=1)
```




    array([[ 0,  1,  2,  0,  2,  4],
           [ 3,  4,  5,  6,  8, 10],
           [ 6,  7,  8, 12, 14, 16]])




```python
#垂直叠加
np.vstack((e,f))
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])




```python
#用concatenate()函数也能达到同样的效果
np.concatenate((e,f),axis=0)
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])




```python
#深度叠加
np.dstack((e,f))#变为三维
```




    array([[[ 0,  0],
            [ 1,  2],
            [ 2,  4]],
    
           [[ 3,  6],
            [ 4,  8],
            [ 5, 10]],
    
           [[ 6, 12],
            [ 7, 14],
            [ 8, 16]]])




```python
#列式叠加
oned=np.arange(2)
oned
```




    array([0, 1])




```python
twice_oned=2*oned
twice_oned
```




    array([0, 2])




```python
np.column_stack((oned,twice_oned))
```




    array([[0, 0],
           [1, 2]])




```python
np.column_stack((e,f))
```




    array([[ 0,  1,  2,  0,  2,  4],
           [ 3,  4,  5,  6,  8, 10],
           [ 6,  7,  8, 12, 14, 16]])




```python
np.column_stack((e,f))==np.hstack((e,f))#用==运算符对两个数组进行对比
```




    array([[ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True]], dtype=bool)




```python
#行式堆叠
np.row_stack((oned,twice_oned))
```




    array([[0, 1],
           [0, 2]])




```python
np.row_stack((e,f))#对于二维数组，row_stack()函数相当于vstack（）函数
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])




```python
np.row_stack((e,f))==np.vstack((e,f))
```




    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)




```python
e
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
#横向拆分
np.hsplit(e,3)
```




    [array([[0],
            [3],
            [6]]), array([[1],
            [4],
            [7]]), array([[2],
            [5],
            [8]])]




```python
#相当于调用参数axis=1的split()函数：
np.split(e,3,axis=1)
```




    [array([[0],
            [3],
            [6]]), array([[1],
            [4],
            [7]]), array([[2],
            [5],
            [8]])]




```python
#纵向拆分
np.vsplit(e,3)
```




    [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]




```python
#相当于调用参数axis=0的split函数：
np.split(e,3,axis=0)
```




    [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]




```python
#深向拆分：
g=np.arange(27).reshape(3,3,3)
g
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],
    
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])




```python
np.dsplit(g,3)
```




    [array([[[ 0],
             [ 3],
             [ 6]],
     
            [[ 9],
             [12],
             [15]],
     
            [[18],
             [21],
             [24]]]), array([[[ 1],
             [ 4],
             [ 7]],
     
            [[10],
             [13],
             [16]],
     
            [[19],
             [22],
             [25]]]), array([[[ 2],
             [ 5],
             [ 8]],
     
            [[11],
             [14],
             [17]],
     
            [[20],
             [23],
             [26]]])]




```python
#numpy数组的属性
h=np.arange(24).reshape(2,12)
h
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])




```python
h.ndim#维度
```




    2




```python
h.size#元素的数量
```




    24




```python
h.itemsize#元素所占的字节数
```




    4




```python
h.nbytes#存储整个数组所需的字节数量
```




    96




```python
h.size*h.itemsize#nbytes的值是itemsize属性值和size属性值之积
```




    96




```python
h.resize(6,4)
h
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])




```python
h.T#T属性的作用与transpose()函数相同
```




    array([[ 0,  4,  8, 12, 16, 20],
           [ 1,  5,  9, 13, 17, 21],
           [ 2,  6, 10, 14, 18, 22],
           [ 3,  7, 11, 15, 19, 23]])




```python
i=np.array([1+1.j,3+2.j])#复数用j表示
i
```




    array([ 1.+1.j,  3.+2.j])




```python
i.real#返回数组的实部
```




    array([ 1.,  3.])




```python
i.imag#返回数组的虚部
```




    array([ 1.,  2.])




```python
i.dtype#如果数组含有复数，那么它的数据类型将自动变为复数类型
```




    dtype('complex128')




```python
i.dtype.str#相当于complex128
```




    '<c16'




```python
j=np.arange(4).reshape(2,2)
j
```




    array([[0, 1],
           [2, 3]])




```python
k=j.flat
k
```




    <numpy.flatiter at 0x6483e80>




```python
for i in k:
    print(i)
```

    0
    1
    2
    3
    


```python
j.flat[2]
```




    2




```python
j.flat[[1,3]]
```




    array([1, 3])




```python
j.flat=7
j
```




    array([[7, 7],
           [7, 7]])




```python
j.flat[[1,3]]=1#返回指定元素
j
```




    array([[7, 1],
           [7, 1]])




```python
#数组的转换
i
```




    array([ 1.+1.j,  3.+2.j])




```python
i.tolist()#将数组或矩阵转成列表
```




    [(1+1j), (3+2j)]




```python
i.astype(int)
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: ComplexWarning: Casting complex values to real discards the imaginary part
      """Entry point for launching an IPython kernel.
    




    array([1, 3])




```python
i.astype('complex')#把数组元素转化成指定类型
```




    array([ 1.+1.j,  3.+2.j])




```python

```
