# G-R-Joint-Optimization
## 版本说明
&emsp;&emsp;当前版本的生成器与识别器代码已完成整理，联合优化部分还未完全整理。<br />
&emsp;&emsp;实验中存在边做边改的情况，因此修改代码为由统一的配置文件启动的方式需要一定的工作量，目前还未完全修改完成，当前版本仅供参考。<br />
<p align="right">2023年3月15日</p>

## 摘要
Joint Optimization for Attention-based Generation and Recognition of Chinese Characters Using Tree Position Embedding

Abstract: Despite the growing interest in Chinese character generation, creating a nonexistent character remains an open challenge. Radical-based Chinese character generation is still a novel task while radical-based Chinese character recognition is more technologically advanced. To fully utilize the knowledge of recognition task, we first propose an attention-based generator. The generator chooses the most relevant radical to generate each zone with an attention mechanism. Then, we present a joint optimization approach to training generation-recognition models, which can help the generator and recognizer learn from each other effectively. The joint optimization is implemented via contrastive learning and dual learning. Considering the symmetry of the generation and recognition, contrastive learning aims to strengthen the performance of the encoder of recognizer and the decoder of generator. Since the generation and recognition tasks can form a closed loop, dual learning feeds the output from one to another as input. Based on the feedback signals generated during the two tasks, we can iteratively update the two models until convergence. Finally, as our model ignores the order information of a sequence, we exploit position embedding to extend the image representation ability and propose tree position embedding to represent the positional information for tree structure captions of Chinese characters. The experimental results in printed and nature scenes show that the proposed method improves the quality of the generating images and increases the recognition accuracy for Chinese characters.

## 训练  
### 快速开启训练  
```
python train.py --model xxx
```
- 必需参数  
    - --model 模型名称  
        范围: ['generator', 'recognizer', 'joint']  
- 可选参数
    - --pre_model 预加载模型路径
    - --batchsize 批次大小
    - --msg 注释信息

### 测试
```
python test.py --model xxx
```
- 必需参数  
    - --model 模型名称  
        范围: ['generator', 'recognizer', 'joint']  
    - --pre_model 预加载模型路径

