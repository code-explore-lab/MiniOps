# ReLU 算子接口文档

函数签名：

```cpp
namespace miniops {
void relu(const float *input, float *output, int size);
}
```

## reduce_sum

签名：
void reduce*sum(const float *input, float *output, int size);
数学：$s=\sum*{i=0}^{size-1} input[i]$

参数约定

- input：输入数组，只读；size>0时必须合法
- output：指向单个float，用来存放求和结果，调用方保证可写
- size：输入元素数量

行为规范

1. size <= 0：output写入0.0，直接返回（空输入行为固定）
2. size == 1：输出等于输入元素本身
3. 浮点数加法无结合律；后续分块/并行版本结果允许微小偏差，使用容差判断
