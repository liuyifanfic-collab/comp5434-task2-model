# 任务二特征生产规范（交 C 执行）

> 本规范定义你需要产出的特征文件：输入、输出、列定义、硬性规则、边界。

## 0. 任务一句话

用 Spark（Hadoop/Flink 亦可）从 train.csv / test.csv 计算 16 列特征，产出两个特征文件交回。

## 1. 输入

| 文件 | 说明 |
|---|---|
| `train.csv` | 768 万条评论，含 rating（Kaggle 下载） |
| `test.csv` | 1 万条评论，无 rating（Kaggle 下载，与上面同一版本） |
| `split/val_ids.csv` | **76.8 万验证折 id 清单（红线文件，见规则 1）** |
| `split/stats.json` | 回退值：训练折全局均值 3.84186 |

后两个文件随本规范一并提供。

## 2. 输出

| 项 | 要求 |
|---|---|
| 路径 | `features/train_features.csv`、`features/test_features.csv`（数据量大可换 parquet） |
| 主键 | `id`，与原始文件一一对应；行序可乱，不可缺失、不可重复 |
| 编码 | UTF-8，含表头 |
| 缺失 | 数值列不允许空值（无统计的行填回退值） |

## 3. 特征列（16 列）

| # | 列名 | 类型 | 计算口径 |
|---|---|---|---|
| 1 | `id` | int | 原样透传 |
| 2 | `user_rating_mean` | float | 该用户在**训练折**全部评论 rating 的均值 |
| 3 | `user_rating_std` | float | 同上标准差；样本 <2 条填 0 且 flag=0 |
| 4 | `user_rating_cnt` | int | 该用户训练折评论条数 |
| 5 | `prod_rating_mean` | float | 按 `prod_id` 在训练折的 rating 均值 |
| 6 | `prod_rating_std` | float | 同上标准差；样本 <2 条填 0 且 flag=0 |
| 7 | `prod_rating_cnt` | int | 该商品训练折评论条数 |
| 8 | `parent_prod_rating_mean` | float | 按 `parent_prod_id` 在训练折的 rating 均值 |
| 9 | `user_text_len_mean` | float | 该用户训练折评论 `comment` 字符数均值 |
| 10 | `time_hour` | int | `time` 转 UTC+8 后的小时（0–23） |
| 11 | `time_dow` | int | `time` 转 UTC+8 后的星期（0–6，周一=0） |
| 12 | `votes` | int | 原样透传 |
| 13 | `purchased` | int | 原样透传 |
| 14 | `user_known_flag` | int | 0/1：第 2–4 列来自真实历史=1，回退=0 |
| 15 | `prod_known_flag` | int | 0/1：第 5–7 列同上 |
| 16 | `parent_known_flag` | int | 0/1：第 8 列同上 |

## 4. 三条硬性规则

1. **防泄漏**：凡带 rating 的统计（第 2–8 列），**只准用训练折行计算**。做法：读入 train.csv 后，按 `val_ids.csv` 的 id 剔除那 76.8 万行，再分组统计
2. **映射 + 回退**：验证折行和 test.csv 全部行的特征，按 `user_id` / `prod_id` 从训练折统计表映射；查不到的（新用户/新商品）填 3.84186，对应 flag=0
3. **别自行重算切分**：切分以 `val_ids.csv` 为准。Spark 侧行序、随机数实现与 pandas 不同，自己按 seed 重算会得到不同的切分 → 泄漏

## 5. 边界：不用你做的

- 文本向量化（TF-IDF）：归 B（柳），无需产出任何文本特征
- 模型训练与预测：归 B（柳）
- 若后期文本计算需 Spark 并行化，另行协商，当前不涉及

## 6. 交付要求

- **期望 v1 交付：11-05 前**
- 交付时附带：生成命令、运行时长（可直接作报告效率指标素材）、行数自检结果
- 交付前自检：
  - [ ] 两个文件行数分别 = 原始 train / test 行数
  - [ ] `id` 无缺失、无重复
  - [ ] 所有 flag=0 的行，统计值 = 3.84186
  - [ ] 数值列无空值
- 后续加列：直接加 + 群里说一声；改名 / 删列 / 改语义：先提出，确认后再动
