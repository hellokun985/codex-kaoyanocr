# 数学题截图结构化识别项目

当前版本是一个按统一数据结构收口后的最小可运行链路，支持以下三类题型：

- `single_choice`
- `fill_blank`
- `solution`

当前主链路：

1. 图像预处理
2. OCR 结果读取
3. 按 `y` 坐标聚类成行
4. 规则分类
5. 按题型解析
6. 输出 `ParsedQuestion` JSON

## 项目结构

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── src/
│   └── math_question_ocr/
│       ├── __init__.py
│       ├── data_models.py
│       ├── image_preprocessor.py
│       ├── line_clusterer.py
│       ├── minimal_pipeline.py
│       ├── ocr_stub.py
│       ├── question_classifier.py
│       ├── rule_classifier.py
│       ├── engines/
│       │   └── __init__.py
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── single_choice_parser.py
│       │   ├── fill_blank_parser.py
│       │   └── solution_question_parser.py
│       └── pipeline/
│           └── __init__.py
├── examples/
│   └── sample_question.ocr.json
└── tests/
    ├── test_question_classifier.py
    ├── test_single_choice_parser.py
    ├── test_fill_blank_parser.py
    └── test_solution_question_parser.py
```

## 安装

只安装基础包：

```bash
pip install -e .
```

安装测试依赖：

```bash
pip install -e ".[dev]"
```

如果需要启用图像预处理探测能力：

```bash
pip install -e ".[vision]"
```

## 输入说明

运行时需要提供：

- 输入图片路径，例如 `samples/question_001.png`
- 可选 OCR sidecar JSON，例如 `samples/question_001.png.ocr.json`
- 可选图形 sidecar JSON，例如 `samples/question_001.png.figures.json`

输入图片路径就是 `main.py` 的第一个位置参数：

```bash
python3 main.py samples/question_001.png
```

## CLI 命令

只传图片路径：

```bash
python3 main.py samples/question_001.png
```

传图片路径和 OCR sidecar JSON：

```bash
python3 main.py samples/question_001.png \
  --ocr-json samples/question_001.png.ocr.json
```

同时传 OCR 和图形 sidecar JSON：

```bash
python3 main.py samples/question_001.png \
  --ocr-json samples/question_001.png.ocr.json \
  --figure-json samples/question_001.png.figures.json
```

指定调试图片输出目录：

```bash
python3 main.py samples/question_001.png \
  --ocr-json samples/question_001.png.ocr.json \
  --preprocess-out ./debug_out
```

## 输出 JSON 示例

`main.py` 会输出一个 JSON 对象，包含：

- `preprocess`
- `ocr_blocks`
- `line_clusters`
- `parsed_question`

示例：

```json
{
  "preprocess": {
    "original_path": "samples/question_001.png",
    "processed_path": "debug_out/question_001.preprocessed.png",
    "width": 1080,
    "height": 1520,
    "metadata": {
      "preprocess_mode": "opencv_gray_otsu"
    }
  },
  "ocr_blocks": [
    {
      "block_id": "b1",
      "text": "1. 已知集合 A={1,2},B={2,3}，则 A∩B=",
      "bbox": {
        "x1": 40,
        "y1": 40,
        "x2": 980,
        "y2": 100
      },
      "confidence": 1.0,
      "block_type": "text",
      "line_id": "l1",
      "source": "sidecar_json",
      "metadata": {}
    }
  ],
  "line_clusters": [
    {
      "line_id": "l1",
      "bbox": {
        "x1": 40,
        "y1": 40,
        "x2": 980,
        "y2": 100
      },
      "blocks": [],
      "text": "1. 已知集合 A={1,2},B={2,3}，则 A∩B=",
      "reading_order": 1
    }
  ],
  "parsed_question": {
    "question_no": "1",
    "score": null,
    "question_type": "single_choice",
    "has_figure": false,
    "figures": [],
    "stem_segments": [
      {
        "region_id": "stem_text_l1",
        "region_type": "stem_text",
        "bbox": {
          "x1": 40,
          "y1": 40,
          "x2": 980,
          "y2": 100
        },
        "text": "已知集合 A={1,2},B={2,3}，则 A∩B=",
        "block_ids": [
          "b1"
        ],
        "reading_order": 1,
        "metadata": {}
      }
    ],
    "stem": "已知集合 A={1,2},B={2,3}，则 A∩B=",
    "options": [],
    "blanks": [],
    "subquestions": [],
    "layout_regions": []
  }
}
```

## 调试图片输出位置

如果命令里传了：

```bash
--preprocess-out ./debug_out
```

那么预处理后的图片会输出到：

```text
./debug_out/<原图文件名去后缀>.preprocessed.png
```

例如输入：

```text
samples/question_001.png
```

则输出图片路径通常为：

```text
./debug_out/question_001.preprocessed.png
```

如果没有传 `--preprocess-out`，则不会写出调试图片文件，只在 JSON 里返回原图路径。

## OCR Sidecar JSON 格式

最小示例：

```json
{
  "blocks": [
    {
      "block_id": "b1",
      "text": "1. 已知集合 A={1,2}, B={2,3}，则 A∩B=",
      "bbox": {"x1": 40, "y1": 40, "x2": 980, "y2": 100},
      "line_id": "l1"
    },
    {
      "block_id": "b2",
      "text": "A. {1}",
      "bbox": {"x1": 70, "y1": 140, "x2": 250, "y2": 190},
      "line_id": "l2"
    }
  ]
}
```

图形 sidecar JSON 示例：

```json
{
  "figures": [
    {
      "region_id": "fig_1",
      "bbox": {"x1": 280, "y1": 130, "x2": 720, "y2": 460},
      "text": "几何图",
      "reading_order": 2,
      "figure_type": "geometry_diagram",
      "caption": "三角形图"
    }
  ]
}
```

## 测试运行方式

运行全部测试：

```bash
python3 -m pytest
```

运行指定测试：

```bash
python3 -m pytest tests/test_question_classifier.py
python3 -m pytest tests/test_single_choice_parser.py
python3 -m pytest tests/test_fill_blank_parser.py
python3 -m pytest tests/test_solution_question_parser.py
```

## 当前范围

当前已实现：

- 统一数据结构
- 规则题型分类
- 单选题解析
- 填空题解析
- 解答题解析
- 最小主链路
- sidecar OCR / figures 输入

当前未接入：

- 真实 PaddleOCR
- 真实公式识别模型
- 真实图形检测模型
- 开放场景复杂版式泛化

## 说明

当前工程以 `src/math_question_ocr/data_models.py` 为唯一核心数据结构来源，`ParsedQuestion` 是统一输出对象。
