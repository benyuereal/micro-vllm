# TODO

## [x] spec_decode.py 临时张量改初始化常驻 GPU 缓冲区
- `core/spec_decode.py` 里所有 `torch.tensor(...)` / `torch.arange(...)` 等**每步创建临时张量**的地方，
  全部移到 `__init__`（或 controller 构建时）一次性创建**常驻 GPU 设备缓冲区**（按 max 尺寸预分配）。
- 每步 forward 只往缓冲区**前缀写入**（`buf[:end] = ...`），kernel/下游用 `buf[:end]` 切片取数据，
  不再每步新建张量（减少 allocator 压力 + 避免 graph 路径下的临时分配）。
- 典型对象：context_ids / context_pos / 位置编码索引 / mask / 采样相关小张量等（以实际代码为准，
  逐个排查 `torch.tensor`、`torch.arange`、`torch.full`、`torch.zeros` 在 forward 热路径里的调用）。
- 注意：缓冲区 dtype/device 与原来一致；`[:end]` 切片是 view 零拷贝；pad 区域内容无所谓（kernel 只读 [:end]）。
- 验证：改完跑 demo/verify_spec_qwen38.py 确认 spec vs 非 spec 仍逐 token 一致。
