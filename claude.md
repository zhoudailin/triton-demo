# Claude Code 项目信息

## FunASR 路径
FunASR 框架代码在 `.venv` 中的路径：
- **主路径**: `.venv/lib64/python3.10/site-packages/funasr/`
- **前端处理模块**: `.venv/lib64/python3.10/site-packages/funasr/frontends/`
- **Wav前端**: `.venv/lib64/python3.10/site-packages/funasr/frontends/wav_frontend.py`

## 项目结构
- `test.py` - 流式FBANK测试代码
- `triton/` - Triton推理服务器配置
- `onnx/` - ONNX模型文件
- `wheel/` - Python wheel包

## 关键文件
- `.venv/lib64/python3.10/site-packages/funasr/frontends/wav_frontend.py` - 流式音频前端处理实现