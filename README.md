# 跳绳动作四分类系统

基于 YOLOv8-pose 和随机森林分类器的跳绳动作识别系统，支持四种跳绳动作分类：

- **not_jump** (0): 未跳跃
- **single_under** (1): 单摇跳
- **double_under** (2): 双摇跳
- **cross_jump** (3): 交叉跳

## 快速使用

### 1. 训练模型
```bash
python train_jump_rope_classifier.py --manifest manifest_multiclass_v1.csv --output models/jump_rope_rf_multiclass_v2.pkl --window-size 30 --stride 10
```

### 2. 预测视频
```bash
python predict_jump_rope_classifier.py --video test.mp4 --classifier models/jump_rope_rf_multiclass_v2.pkl --show
```

### 3. 实时检测（基于规则）
```bash
python main.py
```

## 技术架构

- **姿态估计**: YOLOv8n-pose
- **特征工程**: 25维姿态统计特征（脚踝运动、手腕动态、前臂变化等）
- **分类器**: 随机森林（300棵树，最大深度8）
- **窗口策略**: 30帧滑动窗口，步长10帧
