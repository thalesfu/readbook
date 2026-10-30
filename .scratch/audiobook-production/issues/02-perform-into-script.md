# 02: 表演设置接进剧本

Type: task
Status: open
Blocked by: —

## Question

按合成规格第三节，把 11 号票的表演设置并进正式的剧本流程：

- **新建剧本**：`passage.load_rows` 改成剧本化 → 停顿 → 表演设置（`perform.label`）→ 写剧本，每行加 `perform: {vec, tempo, gain}`。
- **只存三个字段**：
  - `perform.SCHEMA` 和提示词去掉 `strength` 和 `emo`；
  - `why` 要不要留：它可能是让模型先想清楚依据再给 `vec`，去掉可能影响结果。先在一份剧本上比一次再定；
  - 不管留不留，剧本里都只存三个字段。
- **补已有的剧本**：照 `readbook.pauses` 的做法，先 `标注` 出来看，认可了再 `写回`。
- **合成和混音从剧本读**：
  - `index_passage` 从剧本行的 `perform` 读，不再读 `books/wa2/trials/perform/表演-*.jsonl`；
  - `mix.assemble` 从 `perform.gain` 读；
  - 没有 `perform` 的按默认值：情绪按 `mood` 查表，语速 1，音量 0；
  - 讲述模式的行 `vec` 为 0。
- **现有两份剧本补 `perform`**：
  - `02-2011-01-01` 和 `2010-11-25 〈2002〉`；
  - 直接用 11 号票已有的表演设置（`books/wa2/trials/perform/表演-<剧本>.jsonl`）里的 `vec`、`tempo`、`gain` 写回，不重新问 `claude -p`，这样才能和定版对得上。

## 要做的

- **验证**：
  - 小样和〈2002〉改用剧本里的 `perform` 走一遍，不重新合成，复用现成的候选；
  - 成品要和 `2.5-表演B混音` 逐句一样。
- **不用补**：07-2013-01-26 等全书剧本化时再说。
