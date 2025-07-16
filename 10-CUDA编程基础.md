# 10-CUDA编程基础模块

## 模块目标
理解CUDA编程基础，包括GPU硬件架构、编程模型、内存层次和线程组织，为NDGTD的CUDA并行化做准备。

## 1. GPU硬件架构

### 1.1 GPU vs CPU架构对比

#### CPU架构特点
- **少量强大核心**：4-64个复杂核心
- **大缓存层次**：L1/L2/L3缓存
- **复杂控制逻辑**：分支预测、乱序执行
- **高时钟频率**：3-5 GHz
- **适合**：串行计算、复杂逻辑

#### GPU架构特点
- **大量简单核心**：数千个简单核心
- **小缓存层次**：共享内存、寄存器
- **简单控制逻辑**：SIMT执行模型
- **较低时钟频率**：1-2 GHz
- **适合**：并行计算、数据密集型

### 1.2 GPU内存层次

#### 全局内存 (Global Memory)
```cpp
// 全局内存分配
float *d_data;
cudaMalloc(&d_data, size * sizeof(float));

// 全局内存访问
__global__ void kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // 全局内存访问
}
```

**特点**：
- **容量大**：几GB到几十GB
- **延迟高**：400-800个时钟周期
- **带宽高**：几百GB/s
- **所有线程可访问**

#### 共享内存 (Shared Memory)
```cpp
__global__ void kernel() {
    __shared__ float shared_data[256];  // 共享内存声明
    
    int tid = threadIdx.x;
    shared_data[tid] = global_data[tid];  // 从全局内存加载
    
    __syncthreads();  // 同步所有线程
    
    // 使用共享内存进行计算
    float result = shared_data[tid] + shared_data[tid + 1];
}
```

**特点**：
- **容量小**：每块48KB
- **延迟低**：1-2个时钟周期
- **带宽高**：与寄存器相当
- **块内线程共享**

#### 寄存器 (Registers)
```cpp
__global__ void kernel() {
    float local_var = 1.0f;  // 存储在寄存器中
    int loop_counter = 0;    // 存储在寄存器中
    
    for (int i = 0; i < 100; i++) {
        local_var *= 2.0f;  // 寄存器操作
    }
}
```

**特点**：
- **容量最小**：每线程255个寄存器
- **延迟最低**：1个时钟周期
- **带宽最高**：与计算单元直接连接
- **线程私有**

#### 常量内存 (Constant Memory)
```cpp
// 常量内存声明
__constant__ float const_data[1024];

__global__ void kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = const_data[idx % 1024];  // 常量内存访问
}
```

**特点**：
- **只读**：运行时不可修改
- **缓存**：有专门的常量缓存
- **广播**：同一warp内相同地址可广播

### 1.3 计算能力 (Compute Capability)

#### 主要版本
- **CC 6.x**：Pascal架构 (GTX 10系列)
- **CC 7.x**：Volta/Turing架构 (RTX 20系列)
- **CC 8.x**：Ampere架构 (RTX 30系列)
- **CC 9.x**：Hopper架构 (H100等)

#### 关键参数
```cpp
// 查询设备信息
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
```

## 2. CUDA编程模型

### 2.1 线程层次结构

#### 线程组织
```cpp
// 线程层次定义
dim3 blockDim(256, 1, 1);      // 每块256个线程
dim3 gridDim(1024, 1, 1);      // 1024个块

// 内核启动
kernel<<<gridDim, blockDim>>>(args);

// 线程索引计算
__global__ void kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程ID
    int bid = blockIdx.x;                              // 块ID
    int tid_in_block = threadIdx.x;                    // 块内线程ID
}
```

#### 线程层次特点
- **Thread**：最小执行单元
- **Block**：线程组，可共享内存和同步
- **Grid**：块集合，独立执行
- **Warp**：32个线程的执行单元（硬件概念）

### 2.2 内存模型

#### 内存访问模式
```cpp
// 合并访问（高效）
__global__ void coalesced_access(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = data[tid] * 2.0f;  // 连续访问
}

// 非合并访问（低效）
__global__ void non_coalesced_access(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * 2] = data[tid * 2] * 2.0f;  // 跳跃访问
}
```

#### 内存合并条件
- **对齐访问**：地址对齐到32字节边界
- **连续访问**：同一warp内线程访问连续地址
- **对齐偏移**：访问模式可预测

### 2.3 同步机制

#### 块内同步
```cpp
__global__ void block_sync() {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    shared_data[tid] = global_data[tid];
    
    __syncthreads();  // 块内所有线程同步
    
    // 同步后的计算
    if (tid < 128) {
        shared_data[tid] += shared_data[tid + 128];
    }
}
```

#### Warp内同步
```cpp
__global__ void warp_sync() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 使用shfl进行warp内数据交换
    float value = __shfl_sync(0xffffffff, input_value, lane_id ^ 1);
}
```

#### 全局同步
```cpp
// 全局同步需要多次内核调用
__global__ void kernel1() {
    // 第一阶段计算
}

__global__ void kernel2() {
    // 第二阶段计算
}

// 主程序
kernel1<<<grid, block>>>();
cudaDeviceSynchronize();  // 全局同步
kernel2<<<grid, block>>>();
```

## 3. CUDA编程基础

### 3.1 基本语法

#### 函数修饰符
```cpp
// 主机函数（CPU执行）
__host__ void host_function() {
    // CPU代码
}

// 设备函数（GPU执行）
__device__ float device_function(float x) {
    return x * x;
}

// 全局函数（GPU内核）
__global__ void kernel_function(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = device_function(data[tid]);
}

// 主机设备函数（CPU和GPU都可执行）
__host__ __device__ float common_function(float x) {
    return x + 1.0f;
}
```

#### 变量修饰符
```cpp
// 全局内存变量
__device__ float global_var;

// 常量内存变量
__constant__ float const_var[1024];

// 共享内存变量
__shared__ float shared_var[256];

// 寄存器变量（默认）
float local_var;
```

### 3.2 内存管理

#### 内存分配和释放
```cpp
// 全局内存
float *d_data;
cudaMalloc(&d_data, size * sizeof(float));
cudaFree(d_data);

// 主机内存（页锁定）
float *h_pinned;
cudaMallocHost(&h_pinned, size * sizeof(float));
cudaFreeHost(h_pinned);

// 统一内存
float *u_data;
cudaMallocManaged(&u_data, size * sizeof(float));
cudaFree(u_data);
```

#### 内存传输
```cpp
// 主机到设备
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

// 设备到主机
cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

// 设备到设备
cudaMemcpy(d_data2, d_data1, size * sizeof(float), cudaMemcpyDeviceToDevice);

// 异步传输
cudaMemcpyAsync(d_data, h_data, size * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
```

### 3.3 错误处理

#### 错误检查宏
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), 
                      cudaMemcpyHostToDevice));
```

#### 内核错误检查
```cpp
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());  // 检查内核启动错误
CUDA_CHECK(cudaDeviceSynchronize());  // 检查内核执行错误
```

## 4. 性能优化基础

### 4.1 内存优化

#### 共享内存使用
```cpp
__global__ void matrix_multiply(float *A, float *B, float *C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < N / BLOCK_SIZE; tile++) {
        // 协作加载到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * N + tile * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        
        __syncthreads();
        
        // 计算部分和
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

#### 内存合并访问
```cpp
// 优化前：非合并访问
__global__ void bad_access(float *data, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * stride] = data[tid * stride] * 2.0f;  // 跳跃访问
}

// 优化后：合并访问
__global__ void good_access(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = data[tid] * 2.0f;  // 连续访问
}
```

### 4.2 计算优化

#### 线程块大小优化
```cpp
// 计算最优线程块大小
int max_threads_per_block;
cudaDeviceGetAttribute(&max_threads_per_block, 
                      cudaDevAttrMaxThreadsPerBlock, 0);

// 经验法则：128-256个线程/块
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;

kernel<<<grid_size, block_size>>>(args);
```

#### 循环展开
```cpp
// 循环展开优化
__global__ void unrolled_loop(float *data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 展开4次循环
    for (int i = tid; i < N - 3; i += blockDim.x * gridDim.x) {
        data[i] = data[i] * 2.0f;
        data[i + 1] = data[i + 1] * 2.0f;
        data[i + 2] = data[i + 2] * 2.0f;
        data[i + 3] = data[i + 3] * 2.0f;
    }
}
```

### 4.3 流和异步执行

#### 流的使用
```cpp
// 创建流
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 异步执行
kernel1<<<grid, block, 0, stream1>>>(data1);
kernel2<<<grid, block, 0, stream2>>>(data2);

// 同步流
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// 销毁流
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

## 5. 与NDGTD的对应关系

### 5.1 数据结构映射

#### ELEM结构到GPU
```cpp
// CPU版本
typedef struct _ELEM_ {
    double Ex[ND3D], Ey[ND3D], Ez[ND3D];
    double Hx[ND3D], Hy[ND3D], Hz[ND3D];
    // ... 其他字段
} ELEM;

// GPU版本：结构体数组
__device__ ELEM *d_elems;

// 或者：数组结构体（SoA）
__device__ double *d_Ex, *d_Ey, *d_Ez;
__device__ double *d_Hx, *d_Hy, *d_Hz;
```

#### 内存布局优化
```cpp
// 结构体数组（AoS）
ELEM elems[NUM_ELEMS];  // 适合CPU

// 数组结构体（SoA）
double Ex[NUM_ELEMS * ND3D];  // 适合GPU
double Ey[NUM_ELEMS * ND3D];
double Ez[NUM_ELEMS * ND3D];
```

### 5.2 计算模式映射

#### 体积分计算
```cpp
// CPU版本：循环遍历单元
for (int ie = 0; ie < num_elems; ie++) {
    for (int i = 0; i < ND3D; i++) {
        // 计算体积分
    }
}

// GPU版本：每个线程处理一个插值点
__global__ void volume_integral_kernel() {
    int ie = blockIdx.x;  // 单元索引
    int i = threadIdx.x;  // 插值点索引
    
    if (i < ND3D) {
        // 计算体积分
    }
}
```

#### 面积分计算
```cpp
// GPU版本：每个线程处理一个面插值点
__global__ void surface_integral_kernel() {
    int ie = blockIdx.x;     // 单元索引
    int face = blockIdx.y;   // 面索引
    int i = threadIdx.x;     // 面插值点索引
    
    if (i < ND2D) {
        // 计算面积分
    }
}
```

### 5.3 通信模式映射

#### MPI通信到CUDA
```cpp
// CPU版本：MPI通信
MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, neighbor, tag,
             recv_buffer, size, MPI_DOUBLE, neighbor, tag,
             MPI_COMM_WORLD, &status);

// GPU版本：CUDA-aware MPI
MPI_Sendrecv(d_send_buffer, size, MPI_DOUBLE, neighbor, tag,
             d_recv_buffer, size, MPI_DOUBLE, neighbor, tag,
             MPI_COMM_WORLD, &status);
```

## 6. 性能分析工具

### 6.1 NVIDIA Visual Profiler
```bash
# 使用nvprof
nvprof ./program

# 详细分析
nvprof --metrics all ./program

# 时间线分析
nvprof --print-gpu-trace ./program
```

### 6.2 Nsight Systems
```bash
# 系统级性能分析
nsys profile ./program

# 生成报告
nsys export --type sqlite report.qdrep
```

### 6.3 自定义性能分析
```cpp
// 使用CUDA事件计时
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## 7. 学习建议

### 7.1 实践练习
1. **基础练习**：向量加法、矩阵乘法
2. **内存优化**：测试不同内存访问模式
3. **算法实现**：实现简单的数值算法

### 7.2 性能调优
1. **内存带宽测试**：测试内存访问性能
2. **计算密度测试**：测试计算性能
3. **负载均衡测试**：测试不同线程块大小

### 7.3 与NDGTD结合
1. **数据结构分析**：分析NDGTD数据结构在GPU上的布局
2. **算法分析**：识别适合GPU并行的计算部分
3. **通信分析**：设计GPU-aware的通信策略

---

*掌握这些CUDA基础后，您将能够开始NDGTD的GPU并行化工作。* 