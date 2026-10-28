# VLX-Seek 推理服务 API 文档

基于 [om-ai-lab/VLX-Seek](https://github.com/om-ai-lab/VLX-Seek) 1.5-10B 封装的 FastAPI 推理服务,供外部系统通过 HTTP 调用。

- **Base URL**:`http://<服务器IP>:8000`(当前部署:`http://117.50.199.161:8000`)
- **交互式文档(Swagger UI)**:`{Base URL}/docs`,可以直接在浏览器里试调用
- **OpenAPI JSON**:`{Base URL}/openapi.json`

## 鉴权

所有 `/v1/*` 接口需要在请求头中携带 API Key:

```
X-API-Key: <你的密钥>
```

- 密钥值请向部署者索取,不要提交到代码仓库或公开分享。
- `/healthz`、`/readyz` 两个健康检查接口**不需要**鉴权,方便负载均衡器/监控探活。
- 缺失或错误的 Key 会返回 `401 Unauthorized`。

## 通用说明

### 请求格式

所有 `/v1/*` 业务接口均为 `POST`,且是 `multipart/form-data` 格式(图片 + 表单字段),**不是** JSON body。

### 图片输入(二选一,必须且只能提供一种)

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image` | file | 直接上传图片文件 |
| `image_base64` | string | Base64 编码的图片(可带 `data:image/...;base64,` 前缀,会自动去掉) |

- 不支持传图片 URL 远程拉取(安全设计,避免 SSRF),只能二选一上传文件或传 base64。
- 图片大小限制:**15MB**(解码后),超出返回 `400`。
- 整个请求体(图片 + 其它字段)限制:**20MB**,超出在网关层直接返回 `413`。

### 生成参数(多数接口通用,均为可选表单字段)

| 字段 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | int | `2048` | `1` ~ `4096` | 最大生成 token 数 |
| `temperature` | float | `0.0` | `0.0` ~ `2.0` | 采样温度,`0` 表示贪婪解码 |
| `top_p` | float | `1.0` | `>0` ~ `1.0` | nucleus sampling 阈值 |
| `repetition_penalty` | float | `1.0` | `>0` ~ `2.0` | 重复惩罚系数 |

超出范围会返回 `422 Unprocessable Entity`。

### 候选框列表(`bbox_list`)

部分接口需要/支持传入候选区域框,格式为 JSON 字符串(作为表单字段的值),内容是 `[[x1,y1,x2,y2], ...]` 像素坐标数组,例如:

```
bbox_list=[[12.0, 20.0, 180.0, 210.0], [50.0, 60.0, 220.0, 300.0]]
```

- 每个请求最多 **200** 个框,超出返回 `400`。
- 坐标合法性(框是否有正面积、是否需要裁剪到图片范围内)由模型内部校验,不合法会返回 `400`。

### 统一返回结构 `InferenceResponse`

除 `/v1/tasks`、`/healthz`、`/readyz` 外,所有接口成功时都返回这个结构:

```json
{
  "answer": "模型生成的文本,可能包含 <ground>...</ground><objects><obj0></objects> 这类标记",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "orange"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "bbox_list": [[12.0, 20.0, 180.0, 210.0]],
  "visualization_base64": null
}
```

| 字段 | 说明 |
| --- | --- |
| `answer` | 模型原始回答文本 |
| `result_bbox_list` | 模型最终选中/定位到的目标框列表(没有目标框的任务,如 `/v1/vqa`,固定为空数组) |
| `prompt_tokens` / `completion_tokens` | 本次调用的输入/输出 token 数,可用于统计算量 |
| `bbox_list` | 本次请求实际使用的候选框(显式传入的,或 WeDetect 自动生成的);不涉及候选框的接口为 `null` |
| `visualization_base64` | 当 `visualize=true` 且有结果框时,返回画好框的 PNG 图片(base64),否则为 `null` |

### 错误响应

| HTTP 状态码 | 场景 |
| --- | --- |
| `400` | 参数不合法(如图片太大、两种图片来源同传/都没传、JSON 格式错误、候选框非法、WeDetect 没生成框且未关闭 `auto_propose`) |
| `401` | 缺少或错误的 `X-API-Key` |
| `413` | 整个请求体超过 20MB |
| `422` | 字段类型/取值范围不对(FastAPI 自动校验,如 `max_new_tokens` 超限、必填字段缺失) |
| `500` | 服务内部错误 |

错误返回统一是 `{"detail": "错误信息"}`。

---

## 接口列表

| 方法 | 路径 | 对应 VLX-Seek 任务 | 一句话说明 |
| --- | --- | --- | --- |
| POST | `/v1/vqa` | `vqa` | 看图问答,自由提问 |
| POST | `/v1/detect` | `detection` | 检测某类/某几类物体,返回所有匹配框 |
| POST | `/v1/ground` | `grounding_single` | 根据描述定位唯一一个目标 |
| POST | `/v1/count` | `counting` | 数某类目标的数量 |
| POST | `/v1/reasoning-detect` | `reasoning_detection` | 带推理过程的目标检测(比 detect 更复杂的描述) |
| POST | `/v1/region/caption` | `brief_region_caption` | 给指定候选框生成简短描述 |
| POST | `/v1/region/ocr` | `region_ocr` | 识别指定候选框内的文字 |
| POST | `/v1/predict` | 原始 prompt | 直接传 prompt,绕过上面的任务模板 |
| GET | `/v1/tasks` | — | 查询服务支持的任务名/语言列表 |
| GET | `/healthz` | — | 存活检查,固定返回 `{"status": "ok"}` |
| GET | `/readyz` | — | 就绪检查,返回模型是否已加载、使用的设备 |

---

### POST `/v1/vqa` — 看图问答

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `question` | 是 | 自由提问的问题 |
| `max_new_tokens` / `temperature` / `top_p` / `repetition_penalty` | 否 | 见上方通用生成参数 |

```bash
curl -X POST http://117.50.199.161:8000/v1/vqa \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "question=图片里有什么水果?"
```

响应中 `result_bbox_list` 恒为 `[]`,`bbox_list` 恒为 `null`(该任务不涉及候选框)。

---

### POST `/v1/detect` — 目标检测

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `text` | 是 | 要检测的类别,多个类别用 `"; "` 分隔,例如 `"orange; apple"` |
| `lang` | 否,默认 `en` | `en` 或 `zh` |
| `bbox_list` | 否 | 显式候选框(JSON 字符串);不传则走下面的 `auto_propose` |
| `auto_propose` | 否,默认 `true` | 为 `true` 且未传 `bbox_list` 时,服务会自动用内置的 WeDetect 检测器生成候选框 |
| `visualize` | 否,默认 `false` | 为 `true` 时额外返回画框后的图片(base64 PNG) |
| 生成参数 | 否 | 见上方通用生成参数 |

> 若 `bbox_list` 未传且 `auto_propose=false`,会返回 `400`(没有可用的候选框)。

```bash
curl -X POST http://117.50.199.161:8000/v1/detect \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "text=orange; apple" \
  -F "visualize=true"
```

---

### POST `/v1/ground` — 指代定位

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `description` | 是 | 描述唯一目标的指代表达,例如 `"左边最大的那个苹果"` |
| `lang` / `bbox_list` / `auto_propose` / `visualize` / 生成参数 | 否 | 同 `/v1/detect` |

```bash
curl -X POST http://117.50.199.161:8000/v1/ground \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "description=最左边的橙子"
```

---

### POST `/v1/count` — 计数

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `description` | 是 | 要计数的目标描述,例如 `"红色的汽车"` |
| `lang` / `bbox_list` / `auto_propose` / `visualize` / 生成参数 | 否 | 同 `/v1/detect` |

```bash
curl -X POST http://117.50.199.161:8000/v1/count \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "description=苹果"
```

---

### POST `/v1/reasoning-detect` — 推理式检测

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `text` | 是 | 需要推理才能定位的目标描述,例如 `"适合榨汁的水果"` |
| `lang` / `bbox_list` / `auto_propose` / `visualize` / 生成参数 | 否 | 同 `/v1/detect` |

```bash
curl -X POST http://117.50.199.161:8000/v1/reasoning-detect \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "text=哪些水果已经成熟了"
```

---

### POST `/v1/region/caption` — 区域简述

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `bbox_list` | **是** | 候选框列表(JSON 字符串),例如 `[[1,2,10,12],[20,20,60,60]]` |
| `target_region_indexes` | **是** | 要描述哪几个框,是 `bbox_list` 的下标数组(JSON 字符串),例如 `[0, 1]` |
| `lang` / `visualize` / 生成参数 | 否 | 同上 |

> 该接口不走自动候选框生成,`bbox_list` 和 `target_region_indexes` 必须都传,否则 `422`。

```bash
curl -X POST http://117.50.199.161:8000/v1/region/caption \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F 'bbox_list=[[12,20,180,210]]' \
  -F 'target_region_indexes=[0]'
```

---

### POST `/v1/region/ocr` — 区域文字识别

参数与 `/v1/region/caption` 完全一致,只是任务换成 OCR。

```bash
curl -X POST http://117.50.199.161:8000/v1/region/ocr \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@screenshot.jpg" \
  -F 'bbox_list=[[100,50,400,90]]' \
  -F 'target_region_indexes=[0]'
```

---

### POST `/v1/predict` — 原始 Prompt

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` / `image_base64` | 是(二选一) | 图片 |
| `prompt` | 是 | 直接发给模型的完整 prompt,不走上面几个任务的模板包装 |
| `bbox_list` | 否 | 可选的候选框(用于需要区域感知的自定义 prompt) |
| `visualize` / 生成参数 | 否 | 同上 |

```bash
curl -X POST http://117.50.199.161:8000/v1/predict \
  -H "X-API-Key: <你的密钥>" \
  -F "image=@photo.jpg" \
  -F "prompt=请用一句话描述这张图片"
```

---

### GET `/v1/tasks` — 查询支持的任务/语言

无参数,需要鉴权。

```bash
curl http://117.50.199.161:8000/v1/tasks -H "X-API-Key: <你的密钥>"
```

```json
{"tasks": ["detection", "grounding_single", "counting", "brief_region_caption", "region_ocr", "reasoning_detection"], "langs": ["en", "zh"]}
```

---

### GET `/healthz` — 存活检查(无需鉴权)

```bash
curl http://117.50.199.161:8000/healthz
# {"status": "ok"}
```

### GET `/readyz` — 就绪检查(无需鉴权)

```bash
curl http://117.50.199.161:8000/readyz
# {"model_loaded": true, "device": "cuda:0"}
```

---

## Python 调用示例

```python
import requests

BASE_URL = "http://117.50.199.161:8000"
API_KEY = "<你的密钥>"
HEADERS = {"X-API-Key": API_KEY}

# 文件上传方式
with open("photo.jpg", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/v1/detect",
        headers=HEADERS,
        files={"image": f},
        data={"text": "orange; apple", "visualize": "true"},
        timeout=60,
    )
resp.raise_for_status()
result = resp.json()
print(result["answer"])
print(result["result_bbox_list"])

# base64 方式(例如图片已经在内存里,不想落盘)
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    f"{BASE_URL}/v1/vqa",
    headers=HEADERS,
    data={"question": "图片里有什么?", "image_base64": b64},
    timeout=60,
)
print(resp.json()["answer"])
```

> 注意:`multipart/form-data` 下 `requests` 的 `data=` 参数值必须是字符串(`bool`/`list` 要先转成 `"true"`/JSON 字符串),不能直接传 Python 的 `True`/`list` 对象。

## 其它注意事项

- **超时建议**:单次推理一般 1~5 秒,但受 `max_new_tokens`、图片复杂度影响,客户端超时建议设置 ≥ 60 秒。
- **并发**:服务端默认把所有推理请求串行化到一个 worker(`VLX_API_MAX_CONCURRENT_INFERENCE=1`,单卡单模型实例),并发请求会排队而不是报错,调用方不需要自己限流,但高并发场景下延迟会累加。
- **语言**:`lang` 字段目前支持 `en` / `zh`,影响的是模型内部任务模板的语言,不是强制输出语言。
- **交互式调试**:比起拼 `curl`,更推荐先打开 `{Base URL}/docs` 用 Swagger UI 的 "Try it out" 调试参数,确认没问题后再写进代码。
