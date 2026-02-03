import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_rows, n_cols,
    stride_row,  # 行主序存储中，相邻行之间的步长（通常 = n_cols）
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D平铺矩阵加法模板 (行并行)
    - grid = (n_rows,): 每个Program实例(Block)处理矩阵的一行
    - 这是后续实现LayerNorm的理想起点
    """
    # 1. 确定当前Block处理的行索引
    row_idx = tl.program_id(axis=0)
    
    # 2. 边界检查：如果启动的Block数过多，直接返回
    if row_idx >= n_rows:
        return
    
    # 3. 计算当前行在内存中的起始位置（核心！）
    #    公式：行起始偏移 = 行索引 * 每行的步长（字节偏移）
    row_start = row_idx * stride_row
    
    # 4. 为当前行生成列偏移（0, 1, ..., BLOCK_SIZE-1）
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 创建掩码：仅处理该行内有效的列（防止BLOCK_SIZE > n_cols）
    mask = col_offsets < n_cols
    
    # 5. 加载数据：从各自矩阵的对应行加载
    #    - x_ptr + row_start: 定位到X矩阵当前行的开头
    #    - + col_offsets: 在当前行内进行列偏移
    x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask)
    y_vals = tl.load(y_ptr + row_start + col_offsets, mask=mask)
    
    # 6. 计算并存储结果
    output_vals = x_vals + y_vals
    tl.store(output_ptr + row_start + col_offsets, output_vals, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """使用1D平铺的矩阵加法"""
    assert x.shape == y.shape, "输入形状必须相同"
    output = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    
    # 动态设置BLOCK_SIZE：取列数向上对齐到2的幂（性能优化）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 设置一个合理上限（例如1024）
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # 定义网格：每行一个Block
    grid = (n_rows,)
    
    # 调用内核
    add_kernel[grid](
        x, y, output,
        n_rows, n_cols,
        x.stride(0),  # 传递行的步长（对于连续张量，等于n_cols）
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # 根据BLOCK_SIZE调整
    )
    return output

if __name__ == "__main__":
    # 测试
    M, N = 1024, 768  # 可以测试非方阵
    x = torch.ones((M, N), device='cuda')     # 每个元素 = 1
    y = torch.full((M, N), 2.0, device='cuda')  # 每个元素 = 2
    
    result = add(x, y)
    expected = torch.full((M, N), 3.0, device='cuda')  # 1 + 2 = 3
    
    print(f"形状验证: {result.shape == expected.shape}")
    print(f"数值验证 (最大误差): {torch.max(torch.abs(result - expected)).item():.6f}")
    print(f"结果一致: {torch.allclose(result, expected, rtol=1e-5)}")