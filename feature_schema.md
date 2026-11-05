# 任务二特征需求单（B → C，冻结件①草案）

> 定稿后放 `docs/frozen/feature_schema.md`。原则：**只增列不改名**，改列须先通知 B。
> 本文件是任务二两条工作线的唯一接口：C 按此产出特征，B 只读消费，不 import C 的代码。
> 对外交接版：《给C的特征生产规范.md》（纯 C 视角，已剥离 B 侧内容），发 C 用那份。

## 1. 文件与格式

| 项 | 约定 |
|---|---|
| 文件 | `features/train_features.csv`、`features/test_features.csv`（数据量大可换 parquet） |
| 编码 | UTF-8，含表头 |
| 主键 | `id`，与原始 train.csv / test.csv 一一对应，顺序可乱但不可缺 |
| 缺失 | 数值列不允许可空；无统计值的行填回退值（见 §3） |

## 2. 特征列清单（v1）

| 列名 | 类型 | 含义 | 是否依赖 rating |
|---|---|---|---|
| `id` | int | 关联主键 | 否 |
| `user_rating_mean` | float | 用户历史评分均值 | **是** |
| `user_rating_std` | float | 用户历史评分标准差（不足 2 条填 0） | **是** |
| `user_rating_cnt` | int | 用户历史评分数 | **是** |
| `prod_rating_mean` | float | 商品历史评分均值 | **是** |
| `prod_rating_std` | float | 商品历史评分标准差（不足 2 条填 0） | **是** |
| `prod_rating_cnt` | int | 商品历史评分数 | **是** |
| `parent_prod_rating_mean` | float | 父商品历史评分均值 | **是** |
| `user_text_len_mean` | float | 用户历史评论平均长度（字符数） | 否 |
| `time_hour` | int | 评论发布小时（0–23，UTC+8） | 否 |
| `time_dow` | int | 评论发布星期（0–6） | 否 |
| `votes` | int | 原样透传 | 否 |
| `purchased` | int | 原样透传 | 否 |
| `*_known_flag` | int | user/prod/parent_prod 三列 0/1，标记统计是否来自真实历史（回退行=0） | 否 |

## 3. 防泄漏规则（硬约束）

1. **所有依赖 rating 的统计只在训练折计算**。训练折以 B 生成的 **`split/val_ids.csv`** 为准（76.8 万验证折 id，全量 768 万按 rating 分层 90/10，seed=5434）——**C 直接读该文件做排除，不要自行按 seed 重算**（Spark 侧行序与 RNG 实现差异会导致切分不一致，造成泄漏）
2. 验证折 / 测试集的统计特征**按 user_id / prod_id 从训练折映射**，不得用自身标签
3. 未知用户/商品：回退**训练折全局均值 3.84186**（见 `split/stats.json`），并置对应 `*_known_flag = 0`
4. `user_rating_std` 等不足样本量的情况：填中性值（std 填 0），置 flag=0

## 4. 文本向量化归属

**归 B**（本地已实现，见 `src/text_repr.py`：自实现 TF-IDF）。C 无需产出文本特征；后期若文本特征计算成为瓶颈，再协商由 C 在 Spark 侧并行化，接口不变。

## 5. 变更流程

加列：直接加 + 通知 B。改名/删列/改语义：先在群里提出，B 确认后再动。
