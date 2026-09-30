# 配置说明

仓库包装器只保留通用参数；数据集专属参数继续由
`evaluation/research/eval_unified.py` 处理，可以通过重复的 `--extra` 传入。

## 常用参数

```text
evaluate DATASET [DATASET ...]
  --n / --num-tasks N       评测行数
  --start-index I           从零开始的起始行（默认 0）
  --mode direct|refine_summary|return
  --model NAME              服务商模型标识
  --api-key-env ENV_NAME    保存 key 的环境变量名
  --base-url URL             OpenAI 兼容模型端点
  --data-path PATH           覆盖单个数据集的输入
  --data-root PATH           准备好的数据根目录（默认 ./data/files）
  --save-path PATH           结果根目录（默认 runs/<timestamp>）
  --concurrency N            最大并发题数（profile 默认 1；其他情况为 4）
  --profile auto|default|refine-equal
  --extra FLAG               评测器高级参数，可重复
```

例如，直接选择数据集时会自动采用四个核心 benchmark 共用的 `refine-equal` profile：

```bash
python3 evaluate.py BrowseComp --start-index 100 --n 20 \
  --model YOUR_MODEL --base-url http://model.example/v1 \
  --tokenizer-path /path/to/tokenizer \
  --judge-model YOUR_JUDGE --judge-base-url http://judge.example/v1 \
  --judge-api-key-env JUDGE_API_KEY
```

`--api-key-env` 的值是环境变量名，不是 secret。适配器会把该变量复制到评测器
的私有环境中，命令预览里不会出现它的值。`--model` 和 `--base-url` 不是凭据，
可以显示。

`--dry-run` 只显示最终子进程命令，不会访问模型、搜索服务、网页、judge 或 Docker。
真实运行前会检查所选数据路径是否存在。

## 环境变量

| 变量 | 用途 | 何时需要 |
| --- | --- | --- |
| `MODEL_API_KEY`（或 `--api-key-env` 指定的变量） | model SDK | 所有真实 research 评测 |
| `AREX_MODEL_NAME` | 默认 model | 所有真实 research 评测 |
| `AREX_BASE_URL` | 默认模型端点 | 自定义端点时 |
| `AREX_TOKENIZER_PATH` | 本地 token 统计 | unified research 数据集 |
| `JUDGE_API_KEY`（或 `--judge-api-key-env`） | 外部 judge SDK | `refine-equal` profile |
| `SERPER_API_KEY` | `search`、`google_scholar` | research 工具 |
| `JINA_API_KEY` | `visit` | 私有或限流的 Jina |
| `HF_TOKEN` | Hugging Face 下载 | HLE、GAIA 准备 |
| `MLE_BENCH` | MLE-bench Lite | MLE prepare/run |

可选端点变量是 `SERPER_API_URL`、`SERPER_SCHOLAR_API_URL` 和
`JINA_API_URL`。真实值放在本地被忽略的 `.env` 或 secret manager 中。

## 数据路径

可下载数据集默认放在 `data/files/<dataset>/...`：

| 数据集 | 默认输入 |
| --- | --- |
| BrowseComp | `data/files/BrowseComp/browse_comp_test_set.csv` |
| DeepSearch-QA | `data/files/DeepSearch-QA/DSQA-full.csv` |
| HLE | `data/files/HLE/text_items.jsonl` |
| GAIA-2023-validation-text-103 | `data/files/GAIA-2023-validation-text-103/standardized_data.jsonl` |

不传 `--data-root` 时，`data/files/legacy/` 下匹配的旧文件仍会被接受。其他评测器
数据集使用 `data/research/*/config.json` 里的路径，可以用
`python3 evaluate.py download --list` 查看。

多数据集运行传入 `--data-path` 会直接报错，避免把一个文件应用到所有数据集。

## 四个核心数据集的默认参数

`auto` 会为 BrowseComp、GAIA-2023-validation-text-103、HLE 和 DeepSearch-QA
选择 `refine-equal`：1 并发、10 轮、每轮最多 300 次调用、总计最多 1500 次调用、confidence
tiered review，以及相同的 thinking/采样/token/retry 设置。unified backend 的 summary 使用推理模型；
HLE 的专用 solver 也使用推理模型，但上下文和 review 调用由适配器负责。
judge 必须通过 `--judge-model`、`--judge-base-url` 和 `--judge-api-key-env` 从外部指定；
使用 `--profile default` 可以关闭这组配置。HLE 仍由专用 agent/judge 适配器执行，
但沿用同一套生成、预算和重试参数；它的上下文截断和 judge 调用由该适配器负责。

| 参数 | 默认值 |
| --- | --- |
| 并发 / 外层轮数 | 1 / 最多 10 |
| 每轮 / 每题模型调用上限 | 300 / 1500，含 confidence review |
| Confidence 阈值 | ≥ 95 接受；≥ 90 进入中间档 |
| Thinking / 保留 thinking / summary thinking | 均开启 |
| Temperature / top-p / top-k / min-p | 1.0 / 0.95 / 20 / 0.0 |
| Presence / repetition penalty | 1.5 / 1.0 |
| 上下文 / 响应 / review token 上限 | 240000 / 16384 / 4096 |
| Tool-call regeneration / 逻辑调用尝试 / 请求尝试 | 20 / 5 / 10 |
| 整题尝试 / 超时 | 1 / 86400 秒 |
| 上下文更新触发 / 更新次数上限 | 128000 token / 24（unified backend） |

BrowseComp、GAIA 和 DeepSearch-QA 使用 `refine_summary`。HLE 每次进入下一轮前
审查上一轮答案，review 计入同一调用预算；截断后的响应上限也是 16384，并保留
适配器的 262144-token 单请求用量保护。评分协议和 judge 专属重试仍按数据集保留。
HLE 中间轮次放在 `_hle_outer/`，最终结果才写入 `HLE/`；中断后用相同 `--save-path` 续跑。

## 高级参数和重跑

底层评测器支持：

```bash
python3 evaluate.py HLE --n 20 \
  --extra=--hle-max-completion-tokens=32768 \
  --extra=--disable-visit-fallback
```

完整列表：

```bash
python3 evaluation/research/eval_unified.py --help
```

每个 model、mode 和任务范围使用独立的 `--save-path`。共享 profile 会跳过所有已保存的最终题目；
`--profile default` 只跳过已正确的题。完整重跑请换新的结果目录。`--extra` 可以覆盖单项参数，
覆盖后应按实际配置记录评测方式。
