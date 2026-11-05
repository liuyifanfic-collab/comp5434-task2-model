# 给 D：B 侧交付包（2026-11-04）

## 1. 报告章节

`report_B_task2_model.md`（英文终稿）。3.x 模型设计 / 4.x 实现 / 5.x 结果与消融，[TBD] 已全部回填（效率表）。数字与 `experiments.md` 台账一致，直接统稿插入。

## 2. 代码复现指南（对应你的 I-2）

- 位置：`project/task2_model/`，依赖见 `requirements.txt`（另需 `pyarrow`、CPU 版 `torch`）
- 数据前置：`data/train.csv`、`data/test.csv`（Kaggle）+ `features/`（C 交付）+ `split/`（已冻结，**勿重跑 make_split**）
- 复现命令（按序）：

```bash
cd project/task2_model
python -m tests.smoke_test        # 环境自检，无需数据
python -m src.train_final_neural  # 全量训练 ~24 分钟
python -m src.predict_neural      # 生成提交文件，应与 Kaggle v6 (0.77300) 一致
```

- 复现性：seed 5434、确定性词表；预测与 v6 一致
- 跑不通找 B

## 3. Slides 模型侧素材（建议 1-2 页）

**一句话方案**：自实现两阶段模型（线性 Ridge → 神经 EmbeddingBag+MLP），Spark 统计特征融合，零黑盒。

| 关键数字 | 值 |
|---|---|
| val RMSE 演进 | 1.5154（均值）→ 1.4033（线性）→ **1.1706（神经全量）** |
| Kaggle LB | 1.30841 → **0.77300**（五版） |
| 特征贡献 | -0.035（线性时代）→ -0.001（信号吸收） |
| 效率 | 131.3 s/epoch，离线 1442.5 s，推理 0.02 s |

**三个讲点**（各一句话）：
1. 数据×3 零增益的饱和实验 → 论证模型类升级的必要性
2. 留一法防泄漏（相关性 0.778→0.115，值 0.039 RMSE）
3. 结构捷径排查（配对匹配 0%、文本匹配 7.8%）→ 排除投机，差距归因于模型容量

## 4. 最终提交意见（11-14 前手动选定）

**v6（0.77300）+ v5（0.79）**，各选一次。
