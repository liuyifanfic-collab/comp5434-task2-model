# Task 2 Model（B 独占模块）

> COMP5434 Dataset 2：评论评分预测（RMSE）。本目录是 B 的开发空间，
> 最终按组长方案迁入 `TeamName/code/task2_model/`。

## 当前状态（终版，2026-11-04）

- [x] 两阶段模型全部完成：自实现线性 Ridge（TF-IDF + 19 列特征融合）→ 自实现神经模型（EmbeddingBag + MLP，最终方案）
- [x] 全部实验完成：消融、调参 sweep、bigram 负结果、结构捷径排查（见 `experiments.md`）
- [x] **最终成绩：Kaggle 公榜 0.77300（v6，全量神经模型）**，val RMSE 1.1706
- [x] 效率指标：131.3 s/epoch、离线 1442.5 s、推理 0.02 s（`artifacts/neural_final_run.json`）
- [x] 报告章节终稿（`report_B_task2_model.md`）、给 C 反馈（已发）、给 D 交付包（`给D的交付包.md`）
- [x] 合规：零黑盒（PyTorch 仅作张量引擎，训练循环手写）

## 快速开始

```bash
cd project/task2_model

# 1. 环境自检（无需数据）
python -m tests.smoke_test
python -m tests.fusion_test

# 2. 最终复现（需 data/ + features/ + split/，split 已冻结勿重算）：
python -m src.train_final_neural   # 全量训练 ~24 分钟
python -m src.predict_neural       # 生成提交文件（= Kaggle v6, 0.77300）
```

线性阶段与消融实验入口（复现报告表格用）：
```bash
python -m src.run_ablation    # 五组消融表
python -m src.tune_text       # 线性文本调参 sweep
python -m src.train_neural    # 神经 pilot 对比
```

依赖：`numpy`、`pandas`、`pyarrow`、CPU 版 `torch`（见 `requirements.txt`）。

## 目录结构

```
task2_model/
├─ config.py            # 唯一配置源：路径 / 种子 / 超参
├─ src/
│  ├─ data_loader.py    # 读 CSV + schema 校验 + 文本列拼接
│  ├─ text_repr.py      # 自实现 TF-IDF（子线性 tf + 平滑 idf + L2 归一）
│  ├─ sparse.py         # 自实现 CSR：两个 mat-vec 核 = Spark map-reduce 同构
│  ├─ model.py          # 自实现 Ridge + GD，含 early stop 与逐 epoch 计时
│  ├─ evaluate.py       # RMSE + 分层切分（seed=5434）
│  ├─ train.py          # 训练入口（eval 协议合规：TF-IDF 只在训练折拟合）
│  └─ predict.py        # 预测入口 + 提交文件写出 + 推理计时
├─ tests/smoke_test.py  # 合成数据端到端测试
├─ data/                # 放 train.csv / test.csv（不入库）
├─ artifacts/           # 训练产物：词表 / idf / 权重
├─ submissions/         # task2_submission.csv
├─ feature_schema.md    # ★ 给 C 的特征需求单（两线唯一接口）
└─ eval_protocol.md     # ★ 给 D 的评测协议提案
```

## 黑盒合规说明

本模块 **零 sklearn / 零 Spark MLlib 模型调用**：
- 文本表示：TF-IDF 数学公式手工实现（df 统计 → 平滑 idf → 子线性 tf → L2 归一）
- 模型：Ridge 目标函数 + 解析梯度手工实现，全批量 GD 迭代求解
- 稀疏运算：CSR 两个矩阵向量积用 `np.bincount` 归约手写，且与 Spark 的
  map → reduceByKey 范式一一对应（后续 C 分布式化时数学内核不变）

## 与 C / D 的协作

- **→ C**：`feature_schema.md` 即需求单。C 产出 `features/*.csv` 后，
  B 在 `train.py` 侧加特征拼接与融合实验（v1 消融维度）
- **→ D**：`eval_protocol.md` 即协议提案。效率三指标 B 侧已埋点
  （`model.epoch_times_`、train/predict 入口计时），合并管线后由 C 统一采集
