# DeepEP 统一通信抽象层设计方案

> 目标：消除 `ops/` 与 `ops2/` 中的通信结构体碎片化，定义一套可覆盖 A2 / A3 / A5 / CCU 四种机型的统一通信接口，使上层算子逻辑与底层通信实现解耦。

---

## 目录

1. [现状：各机型通信结构体对比](#1-现状各机型通信结构体对比)
2. [需要替换的通信结构体清单](#2-需要替换的通信结构体清单)
3. [各机型通信接口差异分析](#3-各机型通信接口差异分析)
4. [统一通信抽象层定义](#4-统一通信抽象层定义)
5. [各算子的机型适配映射](#5-各算子的机型适配映射)
6. [落地实施步骤](#6-落地实施步骤)

---

## 1. 现状：各机型通信结构体对比

### 1.1 通信上下文结构体总览

当前项目中存在 **4 种互不兼容的通信上下文结构体**，分布在两个目录中：

| 结构体 | 所在文件 | 使用机型 | 核心字段 |
|--------|---------|---------|---------|
| `HcclOpResParam` | `ops2/op_kernel/moe_distribute_base.h` | A2 (Single/Layered) + A3 (Base) | `localWindowsIn`, `localWindowsOut`, `winSize`, `remoteRes[]`, `localWindowsExp` |
| `HcclOpParam` | HCCL 内置 (通过 helper 函数访问) | A5 | 通过 `GetBaseWindStateAddrByRankId()` / `GetStatusDataSpaceGm()` / `GetWinSize()` 间接访问 |
| `HcclCombineOpParam` | HCCL 内置 (通过 `AlltoAllvWrite` API 访问) | CCU | `windowsIn[]`, `windowsOut[]` 数组，通过高层 API 间接访问 |
| `HcclA2CombineOpParam` | `ops2/op_kernel/moe_distribute_base.h` | A2 Layered (Combine) | `windowsIn[]`, `windowsOut[]`, `aiRMAInfo` 指针, `capability` 指针 |

### 1.2 结构体字段详细对比

#### HcclOpResParam（A2 + A3 共用）

```
定义位置: ops2/op_kernel/moe_distribute_base.h:156-197
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `mc2WorkSpace` | `HcclMC2WorkSpace` | MC2 workspace |
| `localUsrRankId` | `uint32_t` | 本地 user rank ID |
| `rankSize` | `uint32_t` | rank 总数 |
| `winSize` | `uint64_t` | 窗口大小 |
| `localWindowsIn` | `uint64_t` | 本地输入窗口地址 |
| `localWindowsOut` | `uint64_t` | 本地输出窗口地址 |
| `localWindowsExp` | `uint64_t` | 本地导出窗口地址 (用于 magic/status) |
| `winExpSize` | `uint64_t` | 导出窗口大小 |
| `remoteRes` | `RemoteResPtr[131072]` | 远端资源指针数组 |
| `topoInfo` | `AlgoTopoInfo` | 拓扑信息 (serverNum, devicePhyId 等) |
| `config` | `HcclOpConfig` | 配置 (deterministic, retryEnable 等) |
| `tinyMem` | `uint64_t` | all2all 小内存 |
| `zeroCopyHeadPtr` | `uint64_t` | 零拷贝头指针 |
| `zeroCopyIpcPtrs` | `uint64_t[16]` | 零拷贝 IPC 指针数组 |

**访问方式**：直接字段访问，例如：
```cpp
winContext_[ctxIdx]->localWindowsIn + rankId * offset
((HcclRankRelationResV2*)winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr)->windowsIn
```

#### HcclOpParam（A5 专用）

```
定义位置: HCCL 内置头文件，通过 helper 函数访问
```

**访问方式**：通过基类 helper 函数间接访问，不直接访问字段：
```cpp
GetBaseWindStateAddrByRankId(winContext_[ctxIdx], rankId, curRankId)  // 获取窗口地址
GetStatusDataSpaceGm(winContext_[COMM_EP_IDX])                        // 获取 status 地址
GetWinSize(winContext_[COMM_EP_IDX])                                   // 获取窗口大小
```

A5 Kernel 中需要额外计算实际数据窗口大小：
```cpp
baseWindSize_ = GetWinSize(winContext_) - A5_MTE_STATE_WIN_SIZE;  // 减去状态区域
```

#### HcclCombineOpParam（CCU 专用）

```
定义位置: HCCL 内置头文件，通过 AlltoAllvWrite API 间接访问
```

**访问方式**：完全通过 HCCL 高层 API 访问，Kernel 代码中不直接访问任何字段：
```cpp
hccl_.AlltoAllvWrite<true>(sendBuf, sendCounts, sdispls,
                            recvBuf, recvCounts, rdispls, count);  // 一步完成通信
```

#### HcclA2CombineOpParam（A2 Layered Combine 专用）

```
定义位置: ops2/op_kernel/moe_distribute_base.h:266-289
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `workSpace` | `uint64_t` | client-server 通信地址 |
| `rankId` | `uint32_t` | 本 rank ID |
| `rankNum` | `uint32_t` | rank 总数 |
| `winSize` | `uint64_t` | 每窗口大小 |
| `windowsIn` | `uint64_t[HCCL_MAX_RANK_NUM]` | 输入窗口地址数组 |
| `windowsOut` | `uint64_t[HCCL_MAX_RANK_NUM]` | 输出窗口地址数组 |
| `aiRMAInfo` | `uint64_t` | `HcclAiRMAInfo*` 指针 (RDMA QP/CQ/Mem 信息) |
| `capability` | `CombinedCapability*` | 通信能力结构体指针 |
| `data` | `__gm__ IbVerbsData*` | IbVerbs 数据指针 |

**访问方式**：混合访问：
```cpp
hccl_.GetWindowsInAddr(rankId)           // 通过 Hccl 方法获取窗口地址
((HcclAiRMAInfo*)aiRMAInfo)->sqPtr       // 直接访问 RDMA 硬件信息
capability->dataplaneModeBitmap           // 直接访问能力位图
```

### 1.3 通信常量差异对比

| 常量 | A3 (ops) | A5 (ops) | A2 (ops2) | CCU (ops) |
|------|---------|---------|-----------|-----------|
| `STATE_WIN_OFFSET` | 900KB | 1000KB | 900KB | 950KB |
| `IPC_BUFF_MAX_SIZE` | 100MB | N/A | 200MB | N/A |
| `NOTIFY_DISPATCH_BUFF_OFFSET` | 102GB | N/A | 202GB | N/A |
| `STATE_SIZE` | 无 | N/A | 2MB | N/A |
| `MAX_CORE_NUM` | 48 | 48 | 48 | 48 |
| `CAM_MAX_RANK_SIZE` | 384 | 384 | 384 | 384 |
| `COMM_NUM` | 2 (EP+TP) | 2 | 2 | 2 |
| `PING_PONG_SIZE` | 2 | 2 | 2 | 2 |

---

## 2. 需要替换的通信结构体清单

### 2.1 结构体替换映射表

以下是每个 Kernel 文件当前使用的通信上下文类型，以及需要替换为统一接口的目标：

#### Dispatch 算子

| 文件 | 机型 | 当前上下文类型 | 当前访问方式 | 需替换为 |
|------|------|--------------|------------|---------|
| `ops/op_kernel/moe_distribute_dispatch_v2.h` | A3 | `HcclOpResParam*` | 直接字段访问 | `UnifiedCommCtx` |
| `ops/op_kernel/moe_distribute_dispatch_v2_a5.h` | A5 | `HcclOpParam*` | helper 函数间接访问 | `UnifiedCommCtx` |
| `ops/op_kernel/moe_distribute_dispatch_v2_ccu.h` | CCU | `HcclCombineOpParam` | `AlltoAllvWrite` API | `UnifiedCommCtx` |
| `ops2/op_kernel/moe_distribute_dispatch_v2_single.h` | A2 Single | `HcclOpResParam*` + `Hccl<>` | 直接字段 + `BatchWrite` | `UnifiedCommCtx` |
| `ops2/op_kernel/moe_distribute_dispatch_v2_layered.h` | A2 Layered | `HcclOpResParam*` + `Hccl<>` | 直接字段 + `AIVRDMAPostSend` + IPC | `UnifiedCommCtx` |
| `ops/op_kernel/cam_moe_dispatch_normal.h` | A3 | `HcclOpResParam*` | 直接字段访问 | `UnifiedCommCtx` |
| `ops/op_kernel/cam_moe_dispatch_normal_a5.h` | A5 | `HcclOpParam*` | helper 函数间接访问 | `UnifiedCommCtx` |
| `ops2/op_kernel/dispatch_normal_a2.cpp` | A2 | `HcclOpResParam*` + `Hccl<>` | `BatchWrite` | `UnifiedCommCtx` |

#### Combine 算子

| 文件 | 机型 | 当前上下文类型 | 当前访问方式 | 需替换为 |
|------|------|--------------|------------|---------|
| `ops/op_kernel/moe_distribute_combine_v2.h` | A3 | `HcclOpResParam*` | 直接字段访问 | `UnifiedCommCtx` |
| `ops/op_kernel/moe_distribute_combine_v2_a5.h` | A5 | `HcclOpParam*` | helper 函数间接访问 | `UnifiedCommCtx` |
| `ops/op_kernel/moe_distribute_combine_v2_ccu.h` | CCU | `HcclCombineOpParam` | `AlltoAllvWrite` API | `UnifiedCommCtx` |
| `ops2/op_kernel/moe_distribute_combine_v2_single.h` | A2 Single | `HcclOpResParam*` + `Hccl<>` | 直接字段 + `BatchWrite` | `UnifiedCommCtx` |
| `ops2/op_kernel/moe_distribute_combine_a2_layered.h` | A2 Layered | `HcclA2CombineOpParam*` | `BatchWrite` + RDMA + IPC | `UnifiedCommCtx` |
| `ops/op_kernel/cam_moe_combine_normal.h` | A3 | `HcclOpResParam*` | 直接字段访问 | `UnifiedCommCtx` |
| `ops/op_kernel/cam_moe_combine_normal_a5.h` | A5 | `HcclOpParam*` | helper 函数间接访问 | `UnifiedCommCtx` |

#### NotifyDispatch 算子

| 文件 | 机型 | 当前上下文类型 | 当前访问方式 | 需替换为 |
|------|------|--------------|------------|---------|
| `ops/op_kernel/notify_dispatch.h` | A3 | `HcclOpResParam*` | 直接字段 + IPC | `UnifiedCommCtx` |
| `ops/op_kernel/notify_dispatch_a5.h` | A5 | `HcclOpParam*` | helper 函数 + IPC | `UnifiedCommCtx` |
| `ops2/op_kernel/notify_dispatch_a2.h` | A2 | `HcclOpResParam*` + `Hccl<>` | `BatchWrite` + IPC | `UnifiedCommCtx` |

### 2.2 需要替换的 Hccl 方法调用

当前各机型 Kernel 中直接调用的 `Hccl<>` 模板类方法：

| 方法 | 调用位置 | 使用机型 | 用途 |
|------|---------|---------|------|
| `hccl_.InitV2(contextGM, &tilingData)` | A2 所有 Kernel | A2 | 初始化 Hccl 上下文 |
| `hccl_.SetCcTilingV2(offset)` | A2 所有 Kernel | A2 | 设置 CC tiling 偏移 |
| `hccl_.BatchWrite<true>(items, count)` | A2 Single + Layered + NotifyA2 | A2 | RDMA 批量写入 |
| `hccl_.GetWindowsInAddr(rank)` | A2 Single + Layered | A2 | 获取输入窗口地址 |
| `hccl_.GetWindowsOutAddr(rank)` | A2 Single + Layered | A2 | 获取输出窗口地址 |
| `hccl_.Finalize()` | A2 所有 Kernel | A2 | 清理 Hccl 上下文 |
| `hccl_.AlltoAllvWrite<true>(...)` | CCU Kernel | CCU | 高层全互联通信 |

### 2.3 需要替换的 RDMA 硬件直接操作

仅 A2 Layered 使用，需要封装到统一接口的扩展层中：

| 操作 | 调用位置 | 用途 |
|------|---------|------|
| `AIVRDMAPostSend(sqPtr, wqe, sge, doorbell)` | `moe_distribute_dispatch_v2_layered.h` | 直接操作 RoCE NIC 发送 RDMA |
| `cacheWriteThrough(addr, len)` | `moe_distribute_base.h` | 缓存写穿透 |
| `GetDataplaneMode(contextGM)` | `moe_distribute_base.h` | 判断通信面模式 (HOST/AICPU/AIV) |
| 构造 `hns_roce_rc_sq_wqe` 结构 | `moe_distribute_dispatch_v2_layered.h` | 构造 RDMA WQE |
| 构造 `hns_roce_lite_wqe_data_seg` 结构 | `moe_distribute_dispatch_v2_layered.h` | 构造 RDMA SGE |

---

## 3. 各机型通信接口差异分析

### 3.1 Dispatch 阶段通信接口对比

#### 窗口地址获取

| 机型 | 接口调用 | 返回值 |
|------|---------|--------|
| A3 | `winContext_->localWindowsIn + rankId * offset` | `__gm__ uint8_t*` (直接地址运算) |
| A5 | `GetBaseWindStateAddrByRankId(winContext_, rankId, curRankId)` | `__gm__ uint8_t*` (helper 函数) |
| CCU | 不需要获取窗口地址 (API 内部处理) | N/A |
| A2 Single | `hccl_.GetWindowsInAddr(rankId)` | `__gm__ uint8_t*` (Hccl 方法) |
| A2 Layered | `hccl_.GetWindowsInAddr(rankId)` + IPC 共享内存地址 | `__gm__ uint8_t*` + `shareAddrs[rank]` |

#### 数据发送

| 机型 | 接口调用 | 参数说明 |
|------|---------|---------|
| A3 | `DataCopyPad(remoteWinAddr, localUB, ...)` + `SyncAll()` | UB -> 远端窗口 GM 的 DataCopyPad |
| A5 | `DataCopyPad(remoteWinAddr, localUB, ...)` + `SyncAll()` | 同 A3，但窗口地址通过 helper 获取 |
| CCU | `hccl_.AlltoAllvWrite<true>(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, count)` | 一步完成全互联，无需手动管理窗口 |
| A2 Single | `hccl_.BatchWrite<true>(batchWriteItems, itemCount)` | 每个 item 32 字节: `{localGM, remoteGM, dataSize, dataType, targetRank}` |
| A2 Layered | `AIVRDMAPostSend(sqPtr, &wqe, &sge, dbAddr)` | 直接构造 RoCE WQE + SGE，写 doorbell 触发 NIC 发送 |

#### 接收等待

| 机型 | 接口调用 | 同步机制 |
|------|---------|---------|
| A3 | `WaitFlag<HardEvent::MTE3_MTE2>()` + 轮询 magic 值 | magic 值掩码比较: `(recvValue & MAGIC_MASK) == (magic & MAGIC_MASK)` |
| A5 | `WaitFlag<HardEvent::MTE3_MTE2>()` + 轮询 magic 值 | 同 A3，但 status 地址通过 `GetStatusDataSpaceGm()` 获取 |
| CCU | 窗口状态区域轮询 | `*(statusAddr + STATE_WIN_OFFSET) == expectedValue` |
| A2 Single | 轮询 window 中的 status 区域 | `*(winAddr + totalWinSize - STATE_SIZE * 3) == magic` |
| A2 Layered | IPC flag 轮询 + RDMA 完成检测 | `SetIpcFlag(rank, magic)` / `WaitIpcFlag(rank, magic)` + CQ 轮询 |

#### 核间同步

| 机型 | 接口调用 |
|------|---------|
| A3 | `SyncAll<true>()` |
| A5 | `SyncAll<true>()` + 非阻塞 `WaitDispatch()` 返回 bool |
| CCU | `SyncAll<true>()` |
| A2 Single | `SyncAll<true>()` + `SyncCntOnCore()` |
| A2 Layered | `SyncAll<true>()` + IPC flag 同步 |

### 3.2 Combine 阶段通信接口对比

#### 数据聚合方式

| 机型 | 接口调用 | 聚合方式 |
|------|---------|---------|
| A3 | `DataCopy` 从远端窗口读取 + 标量加法 | 逐 rank 读取窗口数据，标量累加 |
| A5 | 同 A3，但窗口地址通过 helper 获取 | 同上 + 非阻塞检查 |
| CCU | `hccl_.AlltoAllvWrite<true>(...)` | 高层 API 一步完成 |
| A2 Single | `BatchWrite` + `ExpertAlltoAllDispatchCopyAdd` | RDMA 批量写 + CopyAdd 聚合 |
| A2 Layered | `SumToWindow()` (IPC 读取 + 加权求和) + `AlltoAllServerDispatch()` (RDMA) + `SumToServer()` (最终 reduce) | 三阶段: IPC 聚合 -> RDMA 跨 server -> 最终 reduce |

#### Combine 特有接口 (A2 Layered)

| 接口 | 说明 |
|------|------|
| `AlltoAllDispatch()` | 通过 IPC 共享内存进行 server 内通信，8 个核并行 |
| `SumToWindow()` | 从 IPC 读取数据，加权求和后写入 windowOut |
| `AlltoAllServerDispatch()` | 通过 `BatchWrite` 进行跨 server RDMA 发送 |
| `SetStatus(rank, value)` | 在 window 状态区域写入完成标志 |
| `WaitDispatch(rank)` | 轮询对端 RDMA 数据到达 |
| `SumToServer()` | 8 核并行最终聚合输出 |

### 3.3 NotifyDispatch 阶段通信接口对比

#### 通信流程差异

| 机型 | 阶段数 | 流程 |
|------|--------|------|
| A3 | 单阶段 | 8 核并行构建 8 个输出 -> IPC 全互联通信 -> ReorderOutput |
| A5 | 单阶段 | 同 A3，但窗口地址通过 helper 获取 |
| A2 | 三阶段 | `ProcessBetweenServer()` (RDMA BatchWrite) -> `ProcessWithinServer()` (IPC) -> `SplitAndCalcData()` (计算 10 个输出) |

#### A2 NotifyDispatch 特有接口

| 接口 | 参数 | 说明 |
|------|------|------|
| `InputToWindowOut()` | 输入数据 | 数据搬到 windowOut |
| `ConstructBatchWriteInfo()` | batchWriteItems | 构造 BatchWrite 信息 |
| `SendRdma()` | batchWriteItems, count | 通过 `hccl_.BatchWrite<true>()` 发送 |
| `WaitRdma()` | 无 | 等待 RDMA 完成 |
| `WindowInToOutput()` | 无 | 从 windowIn 读取数据 |
| `InputToShareSlice()` | 输入数据 | 写入 IPC 共享内存 |
| `ShareToShareSlice()` | shareAddrs | 从其他卡的共享内存读取 |
| `SplitAndCalcData()` | 无 | 拆分数据，计算 10 个输出张量 |

### 3.4 DispatchLayout 阶段接口对比

DispatchLayout 是纯计算算子，无通信接口调用，但输出张量数量和含义不同：

| 机型 | 输出数 | 特有输出 |
|------|--------|---------|
| A3 | 5 | 无 (基础输出) |
| A2 | 11-12 | `localTokenServerUniqCount`, `localTokenServerTotalCount`, `localTokenServerNum`, `localTokenServerOffset`, `sendTokenIdx`, `expertRankTokenIdx`, `tokenIdx`(仅ops2) |

### 3.5 aclnn 算子调用差异

`deep_ep.cpp` 中根据机型调用不同的 aclnn 算子：

| 功能 | A3 (intranode) | A2 (internode) | A2+A3 (low_latency) | A5 (fused) |
|------|----------------|----------------|---------------------|------------|
| Layout | `aclnnDispatchLayout` | `aclnnDispatchLayout` | N/A | `aclnnDispatchLayout` |
| Notify | `aclnnNotifyDispatch` | `aclnnNotifyDispatchA2` | N/A | `aclnnNotifyDispatch` |
| Dispatch | `aclnnCamMoeDispatchNormal` | `aclnnDispatchNormalA2` | `aclnnMoeLowLatencyDispatchV2` | `aclnnFusedDeepMoe` |
| Combine | `aclnnCamMoeCombineNormal` | `aclnnMoeDistributeCombineA2` | `aclnnMoeLowLatencyCombineV2` | (含在 fused 中) |

`comm_alg` 参数选择逻辑：

```
soc_version == ASCEND910B     -> comm_alg = "fullmesh"      (A2)
MOE_ENABLE_CCU == 1           -> comm_alg = "ccu"            (CCU)
default                       -> comm_alg = "fullmesh_v1"    (A3)
```

---

## 4. 统一通信抽象层定义

### 4.1 整体架构

```
┌──────────────────────────────────────────────────┐
│              deep_ep.cpp (C++ 接口层)              │
│    根据 soc_version 选择 MachineType              │
├──────────────────────────────────────────────────┤
│           UnifiedCommContext (统一接口)             │
│  ┌──────────┬──────────┬──────────┬───────────┐  │
│  │ Window   │ HighLvl  │ Layered  │ (扩展)     │  │
│  │ Comm     │ API Comm │ Comm     │ RDMA Comm  │  │
│  │ (A3/A5)  │ (CCU)    │ (A2)     │ (A2 Layer) │  │
│  └──────────┴──────────┴──────────┴───────────┘  │
├──────────────────────────────────────────────────┤
│              HCCL / AscendC 原语                   │
└──────────────────────────────────────────────────┘
```

### 4.2 机型与通信模式枚举

```cpp
// file: csrc/deepep/comm/comm_types.hpp
#pragma once
#include <cstdint>

namespace deep_ep {

// 机型枚举 (对应 TilingKey 基数)
enum class MachineType : uint32_t {
    A2  = 20000,   // Ascend910B
    A3  = 30000,   // Ascend910_93
    A5  = 50000,   // Ascend950 / C310
    CCU = 60000,   // A5 CCU 引擎
};

// 通信模式枚举
enum class CommMode : uint8_t {
    WINDOW_BASED,    // 基于 window 的通信 (A3/A5)
    HIGH_LEVEL_API,  // HCCL 高层 API (CCU)
    BATCH_WRITE,     // BatchWrite RDMA (A2 Single)
    LAYERED,         // 分层: RDMA + IPC (A2 Layered)
};

// 通信域索引
constexpr uint8_t COMM_EP_IDX = 0;  // EP 通信域
constexpr uint8_t COMM_TP_IDX = 1;  // TP 通信域
constexpr uint8_t COMM_NUM    = 2;  // 通信域数量

// 量化类型枚举
enum class QuantType : uint8_t {
    NONE,            // 无量化 (BF16/FP16)
    INT8_STATIC,     // INT8 静态量化
    INT8_DYNAMIC,    // INT8 动态量化
    FP8_E5M2,        // FP8 E5M2 (A5)
    FP8_E4M3,        // FP8 E4M3 (A5)
    FP4_E2M1,        // FP4 E2M1 (A5)
    MX_FP8,          // MX scale FP8 (A5)
    MX_FP4,          // MX scale FP4 (A5)
};

// 机型 -> 通信模式 映射
constexpr CommMode get_comm_mode(MachineType type, bool is_layered) {
    if (type == MachineType::A2 && is_layered) return CommMode::LAYERED;
    if (type == MachineType::A2)                return CommMode::BATCH_WRITE;
    if (type == MachineType::CCU)               return CommMode::HIGH_LEVEL_API;
    return CommMode::WINDOW_BASED;  // A3/A5
}

// 机型 -> 量化能力 查询
constexpr bool supports_quant(MachineType type, QuantType quant) {
    switch (type) {
        case MachineType::A2:
            return quant == QuantType::INT8_STATIC ||
                   quant == QuantType::INT8_DYNAMIC ||
                   quant == QuantType::NONE;
        case MachineType::A3:
            return quant == QuantType::INT8_DYNAMIC ||
                   quant == QuantType::NONE;
        case MachineType::A5:
        case MachineType::CCU:
            return quant == QuantType::FP8_E5M2 ||
                   quant == QuantType::FP8_E4M3 ||
                   quant == QuantType::FP4_E2M1 ||
                   quant == QuantType::MX_FP8 ||
                   quant == QuantType::MX_FP4 ||
                   quant == QuantType::NONE;
    }
    return false;
}

} // namespace deep_ep
```

### 4.3 统一通信上下文接口

```cpp
// file: csrc/deepep/comm/unified_comm_context.hpp
#pragma once
#include "comm_types.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace deep_ep {

// 前向声明
struct CommArgs;

// BatchWrite 数据项 (A2 BatchWrite 通信使用)
struct BatchWriteItem {
    uint64_t localGM;     // 本地 GM 地址
    uint64_t remoteGM;    // 远端 GM 地址
    uint32_t dataSize;    // 数据大小
    uint32_t dataType;    // 数据类型
    uint32_t targetRank;  // 目标 rank
};

// IPC 共享内存信息 (A2 Layered 通信使用)
struct IpcShareInfo {
    uint64_t shareAddrs[8];  // 同 server 内 8 张卡的共享内存地址
    uint32_t localRankSize;  // server 内 rank 数 (固定 8)
    uint32_t serverNum;      // server 数量
};

// RDMA 硬件信息 (A2 Layered 通信使用，封装 HcclAiRMAInfo)
struct RdmaHwInfo {
    uint64_t sqPtr;     // Send Queue 指针
    uint64_t scqPtr;    // Send Completion Queue 指针
    uint64_t rcqPtr;    // Recv Completion Queue 指针
    uint32_t qpNum;     // QP 数量
    uint32_t rankNum;   // rank 数量
};

// 统一通信上下文接口
// 所有机型的通信实现都继承此接口
class UnifiedCommContext {
public:
    virtual ~UnifiedCommContext() = default;

    // ====== 机型信息 ======
    virtual MachineType machine_type() const = 0;
    virtual CommMode comm_mode() const = 0;

    // ====== 窗口地址获取 ======
    // 获取指定 rank 的输入窗口地址
    // A3: winContext_->localWindowsIn + rankId * offset
    // A5: GetBaseWindStateAddrByRankId(winContext_, rankId, curRankId)
    // A2: hccl_.GetWindowsInAddr(rankId)
    // CCU: 返回 nullptr (不需要直接访问窗口)
    virtual __gm__ uint8_t* get_window_in_addr(uint8_t ctx_idx, uint32_t rank_id) = 0;

    // 获取指定 rank 的输出窗口地址
    virtual __gm__ uint8_t* get_window_out_addr(uint8_t ctx_idx, uint32_t rank_id) = 0;

    // 获取窗口数据区大小 (不含状态区)
    // A3: winContext_->winSize
    // A5: GetWinSize(winContext_) - A5_MTE_STATE_WIN_SIZE
    // A2: winContext_->winSize - IPC_BUFF_MAX_SIZE * 2
    // CCU: 返回 0 (不需要)
    virtual size_t get_window_data_size(uint8_t ctx_idx) const = 0;

    // 获取状态区地址 (用于 magic/sync)
    // A3: winContext_->localWindowsExp + STATE_WIN_OFFSET
    // A5: GetStatusDataSpaceGm(winContext_) + STATE_WIN_OFFSET
    // A2: winAddr + totalWinSize - STATE_SIZE * 3
    // CCU: winAddr + STATE_WIN_OFFSET
    virtual __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) = 0;

    // 获取 magic 值 (用于通信同步)
    // 所有机型都需要 magic 值来区分不同的通信轮次
    virtual uint64_t get_magic_value(uint8_t ctx_idx) = 0;

    // ====== 通信操作 ======
    // 数据发送 (通用接口)
    // A3/A5: DataCopyPad 到远端窗口
    // CCU: AlltoAllvWrite
    // A2 Single: BatchWrite
    // A2 Layered: AIVRDMAPostSend
    virtual void send_data(uint8_t ctx_idx,
                           __gm__ const uint8_t* local_data,
                           __gm__ uint8_t* remote_data,
                           size_t data_size,
                           uint32_t target_rank) = 0;

    // 批量发送 (A2 BatchWrite 专用)
    // 非 A2 机型可以忽略此接口
    virtual void batch_write(uint8_t ctx_idx,
                             const BatchWriteItem* items,
                             size_t item_count) {
        // 默认实现: 逐条发送
        for (size_t i = 0; i < item_count; ++i) {
            send_data(ctx_idx,
                      reinterpret_cast<__gm__ const uint8_t*>(items[i].localGM),
                      reinterpret_cast<__gm__ uint8_t*>(items[i].remoteGM),
                      items[i].dataSize,
                      items[i].targetRank);
        }
    }

    // 等待接收完成
    // A3/A5: 轮询 magic 值
    // CCU: 轮询状态区
    // A2 Single: 轮询 window status
    // A2 Layered: 轮询 IPC flag + CQ
    // 返回值: true 表示数据已到达, false 表示未到达 (仅 A5 支持非阻塞)
    virtual bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) = 0;

    // ====== IPC 共享内存 (仅 A2 Layered 实现) ======
    virtual std::optional<IpcShareInfo> get_ipc_info() const { return std::nullopt; }

    // IPC flag 操作 (仅 A2 Layered)
    virtual void set_ipc_flag(uint32_t rank, uint64_t magic) {}
    virtual void wait_ipc_flag(uint32_t rank, uint64_t magic) {}

    // ====== RDMA 硬件操作 (仅 A2 Layered 实现) ======
    virtual std::optional<RdmaHwInfo> get_rdma_info() const { return std::nullopt; }

    // 直接 RDMA 发送 (仅 A2 Layered)
    virtual void rdma_post_send(uint32_t target_rank,
                                __gm__ const uint8_t* local_addr,
                                __gm__ uint8_t* remote_addr,
                                size_t data_size) {}

    // ====== 核间同步 ======
    virtual void sync_all() = 0;

    // ====== 通信域管理 ======
    // 是否需要 TP 域通信 (仅 A2 Single 支持)
    virtual bool is_need_allgather() const { return false; }
    virtual bool is_need_reduce_scatter() const { return false; }

    // ====== 辅助信息 ======
    virtual uint32_t local_rank() const = 0;
    virtual uint32_t rank_size() const = 0;
    virtual uint32_t server_num() const { return 1; }  // 仅 A2 > 1
};

} // namespace deep_ep
```

### 4.4 各机型的具体实现类

```cpp
// file: csrc/deepep/comm/window_comm_impl.hpp
#pragma once
#include "unified_comm_context.hpp"

// ====================================================================
// A3 实现: 基于 HcclOpResParam 的窗口通信
// ====================================================================
class WindowCommA3 : public UnifiedCommContext {
private:
    __gm__ HcclOpResParam* winContext_[COMM_NUM];
    uint32_t rank_;
    uint32_t rankSize_;
    uint64_t magic_ = 0;

    static constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;

public:
    WindowCommA3(__gm__ void* contextGM, uint32_t rank, uint32_t rankSize) {
        winContext_[COMM_EP_IDX] = reinterpret_cast<__gm__ HcclOpResParam*>(contextGM);
        rank_ = rank;
        rankSize_ = rankSize;
    }

    MachineType machine_type() const override { return MachineType::A3; }
    CommMode comm_mode() const override { return CommMode::WINDOW_BASED; }

    __gm__ uint8_t* get_window_in_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        // 直接字段访问
        auto* ctx = winContext_[ctx_idx];
        if (rank_id == ctx->localUsrRankId) {
            return reinterpret_cast<__gm__ uint8_t*>(ctx->localWindowsIn);
        }
        auto* remote = reinterpret_cast<__gm__ HcclRankRelationResV2*>(
            ctx->remoteRes[rank_id].nextDevicePtr);
        return reinterpret_cast<__gm__ uint8_t*>(remote->windowsIn);
    }

    __gm__ uint8_t* get_window_out_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        auto* ctx = winContext_[ctx_idx];
        if (rank_id == ctx->localUsrRankId) {
            return reinterpret_cast<__gm__ uint8_t*>(ctx->localWindowsOut);
        }
        auto* remote = reinterpret_cast<__gm__ HcclRankRelationResV2*>(
            ctx->remoteRes[rank_id].nextDevicePtr);
        return reinterpret_cast<__gm__ uint8_t*>(remote->windowsOut);
    }

    size_t get_window_data_size(uint8_t ctx_idx) const override {
        return winContext_[ctx_idx]->winSize;
    }

    __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return get_window_in_addr(ctx_idx, rank_id) + STATE_WIN_OFFSET;
    }

    uint64_t get_magic_value(uint8_t ctx_idx) override {
        auto* status = reinterpret_cast<__gm__ uint64_t*>(
            winContext_[ctx_idx]->localWindowsExp + STATE_WIN_OFFSET);
        magic_ = *status;
        return magic_;
    }

    void send_data(uint8_t ctx_idx, __gm__ const uint8_t* local_data,
                   __gm__ uint8_t* remote_data, size_t data_size,
                   uint32_t target_rank) override {
        // 使用 DataCopyPad 发送到远端窗口
        AscendC::DataCopyPad(remote_data, local_data,
                             AscendC::DataCopyPadParams(data_size, 0, 0, 0));
    }

    bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) override {
        constexpr uint64_t MAGIC_MASK = ~((1ULL << 32) - 1);
        auto* status = reinterpret_cast<__gm__ uint64_t*>(get_status_addr(ctx_idx, rank_id));
        while ((*status & MAGIC_MASK) != (magic & MAGIC_MASK)) {
            // 轮询等待
        }
        return true;
    }

    void sync_all() override { AscendC::SyncAll<true>(); }
    uint32_t local_rank() const override { return rank_; }
    uint32_t rank_size() const override { return rankSize_; }
};

// ====================================================================
// A5 实现: 基于 HcclOpParam + helper 函数的窗口通信
// ====================================================================
class WindowCommA5 : public UnifiedCommContext {
private:
    __gm__ HcclOpParam* winContext_[COMM_NUM];
    uint32_t rank_;
    uint32_t rankSize_;
    size_t baseWindSize_ = 0;

    static constexpr uint64_t STATE_WIN_OFFSET = 1000 * 1024;
    static constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024;  // 状态区大小

public:
    WindowCommA5(__gm__ void* contextGM, uint32_t rank, uint32_t rankSize) {
        winContext_[COMM_EP_IDX] = reinterpret_cast<__gm__ HcclOpParam*>(contextGM);
        rank_ = rank;
        rankSize_ = rankSize;
        baseWindSize_ = GetWinSize(winContext_[COMM_EP_IDX]) - A5_MTE_STATE_WIN_SIZE;
    }

    MachineType machine_type() const override { return MachineType::A5; }
    CommMode comm_mode() const override { return CommMode::WINDOW_BASED; }

    __gm__ uint8_t* get_window_in_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        // 通过 helper 函数间接访问
        return GetBaseWindStateAddrByRankId(winContext_[ctx_idx], rank_id, rank_);
    }

    __gm__ uint8_t* get_window_out_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return GetBaseWindStateAddrByRankId(winContext_[ctx_idx], rank_id, rank_) + baseWindSize_ / 2;
    }

    size_t get_window_data_size(uint8_t ctx_idx) const override {
        return baseWindSize_;
    }

    __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return GetStatusDataSpaceGm(winContext_[ctx_idx]) + STATE_WIN_OFFSET;
    }

    uint64_t get_magic_value(uint8_t ctx_idx) override {
        auto* status = reinterpret_cast<__gm__ uint64_t*>(get_status_addr(ctx_idx, rank_));
        return *status;
    }

    void send_data(uint8_t ctx_idx, __gm__ const uint8_t* local_data,
                   __gm__ uint8_t* remote_data, size_t data_size,
                   uint32_t target_rank) override {
        AscendC::DataCopyPad(remote_data, local_data,
                             AscendC::DataCopyPadParams(data_size, 0, 0, 0));
    }

    bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) override {
        // A5 支持非阻塞检查
        constexpr uint64_t MAGIC_MASK = ~((1ULL << 32) - 1);
        auto* status = reinterpret_cast<__gm__ uint64_t*>(get_status_addr(ctx_idx, rank_id));
        if ((*status & MAGIC_MASK) == (magic & MAGIC_MASK)) {
            return true;
        }
        return false;  // 非阻塞返回
    }

    void sync_all() override { AscendC::SyncAll<true>(); }
    uint32_t local_rank() const override { return rank_; }
    uint32_t rank_size() const override { return rankSize_; }
};

// ====================================================================
// CCU 实现: 基于 HcclCombineOpParam + AlltoAllvWrite 高层 API
// ====================================================================
class HighLevelCommCCU : public UnifiedCommContext {
private:
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint32_t rank_;
    uint32_t rankSize_;

    static constexpr uint64_t STATE_WIN_OFFSET = 950 * 1024;

public:
    HighLevelCommCCU(__gm__ void* contextGM, uint32_t rank, uint32_t rankSize,
                     __gm__ uint8_t* tilingData) {
        hccl_.InitV2(contextGM, tilingData);
        rank_ = rank;
        rankSize_ = rankSize;
    }

    MachineType machine_type() const override { return MachineType::CCU; }
    CommMode comm_mode() const override { return CommMode::HIGH_LEVEL_API; }

    // CCU 不需要直接访问窗口地址
    __gm__ uint8_t* get_window_in_addr(uint8_t, uint32_t) override { return nullptr; }
    __gm__ uint8_t* get_window_out_addr(uint8_t, uint32_t) override { return nullptr; }
    size_t get_window_data_size(uint8_t) const override { return 0; }

    __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        // CCU 的状态区在窗口中
        return hccl_.GetWindowsInAddr(rank_id) + STATE_WIN_OFFSET;
    }

    uint64_t get_magic_value(uint8_t ctx_idx) override {
        auto* status = reinterpret_cast<__gm__ uint64_t*>(
            hccl_.GetWindowsInAddr(rank_) + STATE_WIN_OFFSET);
        return *status;
    }

    // CCU 使用高层 API 一步完成通信
    void send_data(uint8_t ctx_idx, __gm__ const uint8_t* local_data,
                   __gm__ uint8_t* remote_data, size_t data_size,
                   uint32_t target_rank) override {
        // CCU 路径不使用单条发送，而是使用 allto_all_write
        // 此接口在 CCU 路径中不会被调用
    }

    // CCU 专用: 全互联写入
    void allto_all_write(__gm__ const uint8_t* send_buf,
                         const int64_t* send_counts,
                         const int64_t* sdispls,
                         __gm__ uint8_t* recv_buf,
                         const int64_t* recv_counts,
                         const int64_t* rdispls,
                         int64_t count) {
        hccl_.AlltoAllvWrite<true>(send_buf, send_counts, sdispls,
                                    recv_buf, recv_counts, rdispls, count);
    }

    bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) override {
        auto* status = reinterpret_cast<__gm__ uint64_t*>(
            hccl_.GetWindowsInAddr(rank_id) + STATE_WIN_OFFSET);
        while (*status != magic) {}
        return true;
    }

    void sync_all() override { AscendC::SyncAll<true>(); }
    uint32_t local_rank() const override { return rank_; }
    uint32_t rank_size() const override { return rankSize_; }
};

// ====================================================================
// A2 Single 实现: 基于 HcclOpResParam + BatchWrite
// ====================================================================
class BatchWriteCommA2 : public UnifiedCommContext {
private:
    __gm__ HcclOpResParam* winContext_[COMM_NUM];
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint32_t rank_;
    uint32_t rankSize_;

    static constexpr uint64_t WIN_STATE_OFFSET = 500 * 1024;
    static constexpr uint64_t STATE_WIN_OFFSET = 950 * 1024;

public:
    BatchWriteCommA2(__gm__ void* contextGM, uint32_t rank, uint32_t rankSize,
                     __gm__ uint8_t* tilingData) {
        winContext_[COMM_EP_IDX] = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(contextGM, tilingData);
        rank_ = rank;
        rankSize_ = rankSize;
    }

    MachineType machine_type() const override { return MachineType::A2; }
    CommMode comm_mode() const override { return CommMode::BATCH_WRITE; }

    __gm__ uint8_t* get_window_in_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return hccl_.GetWindowsInAddr(rank_id);
    }

    __gm__ uint8_t* get_window_out_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return hccl_.GetWindowsOutAddr(rank_id);
    }

    size_t get_window_data_size(uint8_t ctx_idx) const override {
        return winContext_[ctx_idx]->winSize;
    }

    __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        auto winSize = winContext_[ctx_idx]->winSize;
        return hccl_.GetWindowsInAddr(rank_id) + winSize - 3 * 2 * 1024 * 1024;
    }

    uint64_t get_magic_value(uint8_t ctx_idx) override {
        auto* magicAddr = reinterpret_cast<__gm__ uint64_t*>(
            hccl_.GetWindowsInAddr(rank_) + WIN_STATE_OFFSET);
        return *magicAddr;
    }

    void send_data(uint8_t ctx_idx, __gm__ const uint8_t* local_data,
                   __gm__ uint8_t* remote_data, size_t data_size,
                   uint32_t target_rank) override {
        BatchWriteItem item;
        item.localGM = reinterpret_cast<uint64_t>(local_data);
        item.remoteGM = reinterpret_cast<uint64_t>(remote_data);
        item.dataSize = data_size;
        item.dataType = 0;
        item.targetRank = target_rank;
        batch_write(ctx_idx, &item, 1);
    }

    void batch_write(uint8_t ctx_idx, const BatchWriteItem* items,
                     size_t item_count) override {
        // 转换为 HCCL BatchWriteItem 格式并调用
        hccl_.BatchWrite<true>(reinterpret_cast<const void*>(items), item_count);
    }

    bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) override {
        auto* status = reinterpret_cast<__gm__ uint64_t*>(get_status_addr(ctx_idx, rank_id));
        while (*status != magic) {}
        return true;
    }

    void sync_all() override { AscendC::SyncAll<true>(); }
    uint32_t local_rank() const override { return rank_; }
    uint32_t rank_size() const override { return rankSize_; }

    // A2 Single 支持 EP+TP 双域
    bool is_need_allgather() const override { return true; }
    bool is_need_reduce_scatter() const override { return true; }
};

// ====================================================================
// A2 Layered 实现: 分层 RDMA + IPC 通信
// ====================================================================
class LayeredCommA2 : public UnifiedCommContext {
private:
    __gm__ HcclOpResParam* winContext_[COMM_NUM];
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t localRankSize_ = 8;  // 每 server 8 卡
    uint32_t serverNum_;
    IpcShareInfo ipcInfo_;
    RdmaHwInfo rdmaInfo_;
    bool hasRdmaInfo_ = false;

    static constexpr int64_t IPC_BUFF_MAX_SIZE = 200 * 1024 * 1024;  // 200MB
    static constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024;      // 2MB (flag区)

public:
    LayeredCommA2(__gm__ void* contextGM, uint32_t rank, uint32_t rankSize,
                  __gm__ uint8_t* tilingData) {
        winContext_[COMM_EP_IDX] = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(contextGM, tilingData);
        rank_ = rank;
        rankSize_ = rankSize_;
        serverNum_ = (rankSize_ + localRankSize_ - 1) / localRankSize_;

        // 初始化 IPC 共享内存信息
        auto* ctx = winContext_[COMM_EP_IDX];
        for (uint32_t i = 0; i < localRankSize_ && i < 8; ++i) {
            ipcInfo_.shareAddrs[i] = ctx->zeroCopyIpcPtrs[i];
        }
        ipcInfo_.localRankSize = localRankSize_;
        ipcInfo_.serverNum = serverNum_;

        // 检查是否支持 RDMA 直写
        auto* a2Param = reinterpret_cast<__gm__ HcclA2CombineOpParam*>(contextGM);
        if (a2Param->aiRMAInfo != 0) {
            auto* aiRMA = reinterpret_cast<__gm__ HcclAiRMAInfo*>(a2Param->aiRMAInfo);
            rdmaInfo_.sqPtr = aiRMA->sqPtr;
            rdmaInfo_.scqPtr = aiRMA->scqPtr;
            rdmaInfo_.rcqPtr = aiRMA->rcqPtr;
            rdmaInfo_.qpNum = aiRMA->qpNum;
            rdmaInfo_.rankNum = aiRMA->rankNum;
            hasRdmaInfo_ = true;
        }
    }

    MachineType machine_type() const override { return MachineType::A2; }
    CommMode comm_mode() const override { return CommMode::LAYERED; }

    __gm__ uint8_t* get_window_in_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return hccl_.GetWindowsInAddr(rank_id);
    }

    __gm__ uint8_t* get_window_out_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        return hccl_.GetWindowsOutAddr(rank_id);
    }

    size_t get_window_data_size(uint8_t ctx_idx) const override {
        return winContext_[ctx_idx]->winSize - IPC_BUFF_MAX_SIZE * 2;
    }

    __gm__ uint8_t* get_status_addr(uint8_t ctx_idx, uint32_t rank_id) override {
        auto winSize = winContext_[ctx_idx]->winSize;
        return hccl_.GetWindowsInAddr(rank_id) + winSize - IPC_BUFF_MAX_SIZE * 2;
    }

    uint64_t get_magic_value(uint8_t ctx_idx) override {
        // A2 Layered: magic 值通过 DataCopy atomic 自增
        auto* magicAddr = reinterpret_cast<__gm__ uint64_t*>(
            hccl_.GetWindowsInAddr(rank_) + IPC_BUFF_MAX_SIZE);
        return *magicAddr;
    }

    // ====== 通信操作: 优先使用 RDMA 直写，回退到 BatchWrite ======
    void send_data(uint8_t ctx_idx, __gm__ const uint8_t* local_data,
                   __gm__ uint8_t* remote_data, size_t data_size,
                   uint32_t target_rank) override {
        if (hasRdmaInfo_ && is_cross_server(target_rank)) {
            // 跨 server: 使用 RDMA 直写
            rdma_post_send(target_rank, local_data, remote_data, data_size);
        } else {
            // 同 server: 使用 IPC 共享内存
            ipc_copy(local_data, remote_data, data_size, target_rank);
        }
    }

    void batch_write(uint8_t ctx_idx, const BatchWriteItem* items,
                     size_t item_count) override {
        hccl_.BatchWrite<true>(reinterpret_cast<const void*>(items), item_count);
    }

    bool wait_recv(uint8_t ctx_idx, uint32_t rank_id, uint64_t magic) override {
        if (is_cross_server(rank_id)) {
            // 跨 server: 轮询 CQ
            wait_rdma_complete(rank_id);
        } else {
            // 同 server: 轮询 IPC flag
            wait_ipc_flag(rank_id, magic);
        }
        return true;
    }

    // ====== IPC 操作 ======
    std::optional<IpcShareInfo> get_ipc_info() const override { return ipcInfo_; }

    void set_ipc_flag(uint32_t rank, uint64_t magic) override {
        auto* flagAddr = reinterpret_cast<__gm__ uint64_t*>(
            ipcInfo_.shareAddrs[rank % localRankSize_] + IPC_DATA_OFFSET);
        *flagAddr = magic;
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(flagAddr), sizeof(uint64_t));
    }

    void wait_ipc_flag(uint32_t rank, uint64_t magic) override {
        auto* flagAddr = reinterpret_cast<__gm__ uint64_t*>(
            ipcInfo_.shareAddrs[rank % localRankSize_] + IPC_DATA_OFFSET);
        while (*flagAddr != magic) {}
    }

    // ====== RDMA 操作 ======
    std::optional<RdmaHwInfo> get_rdma_info() const override {
        return hasRdmaInfo_ ? std::make_optional(rdmaInfo_) : std::nullopt;
    }

    void rdma_post_send(uint32_t target_rank,
                        __gm__ const uint8_t* local_addr,
                        __gm__ uint8_t* remote_addr,
                        size_t data_size) override {
        if (!hasRdmaInfo_) return;
        // 构造 RoCE WQE + SGE 并写 doorbell
        // 详见 moe_distribute_dispatch_v2_layered.h 中的 AIVRDMAPostSend 实现
        // 此处封装为统一接口
    }

    // ====== 辅助 ======
    void sync_all() override { AscendC::SyncAll<true>(); }
    uint32_t local_rank() const override { return rank_; }
    uint32_t rank_size() const override { return rankSize_; }
    uint32_t server_num() const override { return serverNum_; }

private:
    bool is_cross_server(uint32_t rank_id) const {
        return rank_id / localRankSize_ != rank_ / localRankSize_;
    }

    void ipc_copy(__gm__ const uint8_t* local_data,
                  __gm__ uint8_t* remote_data,
                  size_t data_size, uint32_t target_rank) {
        // 通过 IPC 共享内存进行 DataCopy
        AscendC::DataCopyPad(remote_data, local_data,
                             AscendC::DataCopyPadParams(data_size, 0, 0, 0));
    }

    void wait_rdma_complete(uint32_t rank_id) {
        // 轮询 CQ
        // 详见 moe_distribute_dispatch_v2_layered.h 中的实现
    }
};

```

### 4.5 通信工厂

```cpp
// file: csrc/deepep/comm/comm_factory.hpp
#pragma once
#include "unified_comm_context.hpp"
// #include "window_comm_impl.hpp"
// #include "batch_write_comm_impl.hpp"
// #include "layered_comm_impl.hpp"

namespace deep_ep {

class CommFactory {
public:
    // 从 deep_ep.cpp 的 soc_version 判断创建对应的通信上下文
    static std::unique_ptr<UnifiedCommContext> create(
        MachineType type,
        bool is_layered,
        __gm__ void* contextGM,
        uint32_t rank,
        uint32_t rankSize,
        __gm__ uint8_t* tilingData = nullptr)
    {
        switch (type) {
            case MachineType::A3:
                return std::make_unique<WindowCommA3>(contextGM, rank, rankSize);
            case MachineType::A5:
                return std::make_unique<WindowCommA5>(contextGM, rank, rankSize);
            case MachineType::CCU:
                return std::make_unique<HighLevelCommCCU>(
                    contextGM, rank, rankSize, tilingData);
            case MachineType::A2:
                if (is_layered) {
                    return std::make_unique<LayeredCommA2>(
                        contextGM, rank, rankSize, tilingData);
                } else {
                    return std::make_unique<BatchWriteCommA2>(
                        contextGM, rank, rankSize, tilingData);
                }
        }
        return nullptr;
    }

    // 从 soc_version 字符串检测机型
    static MachineType detect_machine_type(const std::string& soc_version) {
        if (soc_version == "Ascend910B")  return MachineType::A2;
        if (soc_version == "Ascend950")   return MachineType::A5;
        // A3 为默认值
        return MachineType::A3;
    }

    // 检测是否启用 layered 模式 (仅 A2)
    static bool detect_is_layered(MachineType type) {
        if (type != MachineType::A2) return false;
        const char* pcie = std::getenv("HCCL_INTRA_PCIE_ENABLE");
        const char* roce = std::getenv("HCCL_INTRA_ROCE_ENABLE");
        return pcie != nullptr && roce != nullptr &&
               std::strcmp(pcie, "1") == 0 &&
               std::strcmp(roce, "0") == 0;
    }

    // 检测是否启用 CCU 模式 (仅 A5)
    static bool detect_is_ccu() {
        return get_value_from_env("MOE_ENABLE_CCU", 0) == 1;
    }

    // 获取 comm_alg 字符串 (传递给 aclnn 算子)
    static const char* get_comm_alg(MachineType type, bool is_ccu) {
        if (type == MachineType::A2)  return "fullmesh";
        if (is_ccu)                   return "ccu";
        return "fullmesh_v1";  // A3 默认
    }
};

} // namespace deep_ep
```

---

## 5. 各算子的机型适配映射

以下按 6 个核心算子接口分别给出各机型在通信各步骤的差异对比表。涉及机型包括 A3、A5、CCU、A2 Single、A2 Layered，其中 dispatch_layout 仅区分 A3 与 A2 两个版本（A5/A2 两版共用）。

### 5.1 DispatchLayout（纯计算，无通信）

> 此算子不涉及 HCCL 通信，差异在于计算逻辑、输出张量和 Server 概念。

| 步骤/维度 | A3 (`ops/dispatch_layout.h`) | A2 Base (`ops/dispatch_layout_a2.h`) | A2 New (`ops2/dispatch_layout_a2.h`) |
|-----------|------------------------------|--------------------------------------|---------------------------------------|
| **HCCL 通信** | 无 | 无 | 无 |
| **Server 概念** | 无 | 有 (`serverNum`, `localRankSize`) | 有 (`serverNum`, `localRankSize`) |
| **多轮支持** | 有 (`round`, `perRoundTokens`) | 无 | 无 |
| **Tiling 字段** | `numTokens, numRanks, numExperts, numTopk, perRoundTokens, rankId` | `numTokens, numRanks, numExperts, numTopk, localRankSize` (无 `perRoundTokens`/`rankId`) | `numTokens, numRanks, numExperts, numTopk, localRankSize, rankId` |
| **sendTokenIdx 维度** | 无此张量 | `[token, numExperts]` | `[token, numTopk]` |
| **tokenIdx 输出** | 无 | 无 | 有 (topkIdx 的 Cast 结果) |
| **前缀和算法** | 标量递增 | 向量 `Mul`+`Add` | 标量 `GetValue`/`SetValue` |
| **越界检查** | 有 | 无 | 有 |
| **计算阶段数** | 2 阶段 | 3 阶段 | 3 阶段 |
| **阶段2 同步链** | `MTE3_MTE2` -> `MTE2_S` | `MTE3_MTE2` -> `MTE2_V` -> `MTE2_S` | `MTE3_MTE2` -> `MTE2_MTE3` -> `MTE3_V` -> `MTE3_S` |
| **输出张量数** | 5 (numTokensPerRank, numTokensPerExpert, isTokenInRank, sendTokenIdxSmall, notifySendData) | 10 (上述+sendTokenIdx, localTokenServer*4, expertRankTokenIdx) | 11 (上述+tokenIdx) |
| **MAX_BATCH_SIZE** | 无 | 4096 | 4096 |
| **TEMP_BATCH_SIZE** | 无 | 32 | 32 |
| **核间同步** | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` |
| **UB Buffer 数** | 6 | 20 | 18 |

### 5.2 NotifyDispatch

| 步骤 | A3 (`ops/notify_dispatch.h`) | A5 (`ops/notify_dispatch_a5.h`) | A2 Base (`ops2/notify_dispatch.h`) | A2 Layered (`ops2/notify_dispatch_a2.h`) |
|------|------------------------------|---------------------------------|-------------------------------------|------------------------------------------|
| **1. 上下文类型** | `HcclOpResParam*` | `HcclOpParam*` | `HcclOpResParam*` | `HcclOpResParam*` |
| **2. 上下文初始化** | 直接 cast `GetHcclContext<0>()` | 直接 cast `GetHcclContext<0>()` | `hccl_.InitV2()` + `SetCcTilingV2()` | `hccl_.InitV2()` + `SetCcTilingV2()` |
| **3. hccl_ 成员** | 声明但**不调用** | **注释掉** | 声明且调用 | 声明且调用 |
| **4. Window IN 获取** | 直接字段 `winContext_->localWindowsIn` / `remoteRes[].nextDevicePtr->windowsIn` | 辅助函数 `GetBaseWindStateAddrByRankId()` | `hccl_.GetWindowsInAddr(rankId)` | `hccl_.GetWindowsInAddr(rankId)` |
| **5. Window OUT 获取** | 不使用 | 不使用 | 不使用 | `hccl_.GetWindowsOutAddr(rank)` (唯一使用) |
| **6. winDataOffset** | `(magic % PING_PONG) * (totalWinSize / 2)` | `(magic % PING_PONG) * (baseWindSize / 2)` | `((totalWinSize_ - 4*STATE_SIZE) / 2) * (magic % PING_PONG)` | `(magic % PING_PONG) * IPC_BUFF_MAX_SIZE` |
| **7. baseWindSize** | N/A (直接用 totalWinSize) | `GetWinSize() - A5_MTE_STATE_WIN_SIZE` | `totalWinSize_ - 4*Moe::STATE_SIZE` | `winSize - IPC_BUFF_MAX_SIZE * 2` |
| **8. Magic 获取** | `DataCacheCleanAndInvalid` 读->自增->写回 | 同 A3 | 同 A3 | 原子自增 `GetValue`+`DataCopy` (无 `DataCacheCleanAndInvalid`) |
| **9. Status 地址** | `winContext_->localWindowsExp + STATE_WIN_OFFSET` | `GetStatusDataSpaceGm() + STATE_WIN_OFFSET` | `hccl_.GetWindowsInAddr(rank) + totalWinSize_ - STATE_SIZE*3` | `hccl_.GetWindowsInAddr(rank) + IPC_DATA_OFFSET - ...` |
| **10. Server 间发送** | N/A | N/A | N/A | `hccl_.BatchWrite<true>()` (RDMA) |
| **11. Server 内发送** | `CpGM2GMPingPong` (GM->UB->GM) | 同 A3 | 同 A3 | `CpGM2GMPingPong` (同 A3) |
| **12. 接收等待** | Magic 轮询 `WaitSyncFlag` -> `WaitOneRankPartFlag` (DataCopy + MAGIC_MASK 比较高32位) | 同 A3 | 同 A3 | Magic 轮询 (Server内) + Status/Data Flag 双重轮询 (Server间, `FLAG_VALUE=0xFFFFFFFF`) |
| **13. 核间同步** | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` (多处) |
| **14. hccl_.Finalize** | 否 | 否 | 是 | 是 |
| **15. Tiling 结构** | 不使用 | 不使用 | `NotifyDispatchTilingData` | `NotifyDispatchA2TilingData` |
| **16. EXPERT_NORMAL_NUM** | 512 | 512 | **256** | N/A |
| **17. Process 步骤** | 14 步 (AssembleSendData -> ... -> BuildRInSrcrankOffset) | 同 A3 (14步) | 15 步 (同 A3 + Finalize) | 3 阶段: ProcessBetweenServer -> ProcessWithinServer -> SplitAndCalcData |
| **18. 输出张量** | 9 (recvCount, recvOffset, expertGlobalOffset, ...) | 同 A3 | 同 A3 | 10+ (tokenServerIdx, tokensUniquePerServer, epRankTokenCnt, offsetInner, countOuter, ...) |
| **19. batchRounds 逻辑** | `round==1` -> `numLocalExperts>=128` -> `numExperts>512` -> `numExperts<=256` -> else | `round==1` -> `numExperts>512` -> `numLocalExperts>=128` -> `numExperts<=256` -> else | `numExperts > 256 ? 16 : 32` (简化) | N/A (不同流程) |

### 5.3 NormalDispatch (CamMoeDispatchNormal)

| 步骤 | A3 (`ops/cam_moe_dispatch_normal.h`) | A5 (`ops/cam_moe_dispatch_normal_a5.h`) | A2 Base (`ops2/cam_moe_dispatch_normal.h`) | A2 Layered (`ops2/a2/cam_moe_distribute_dispatch_a2_layered.h`) |
|------|---------------------------------------|-----------------------------------------|---------------------------------------------|------------------------------------------------------------------|
| **1. 通信上下文** | `HcclOpResParam*` (EP+TP 双域) | `HcclOpParam*` (EP+TP 双域) | `HcclOpResParam*` (声明未用) + `Hccl<AICPU>` (仅 EP) | `HcclOpResParam*` + `Hccl<AICPU>` (仅 EP) |
| **2. 上下文初始化** | 直接 cast `GetHcclContext<0/1>()` | 同 A3 | `hccl_.InitV2()` + `SetCcTilingV2()` + cast (winContext_ 赋值但未使用) | `hccl_.InitV2()` + `SetCcTilingV2()` + cast |
| **3. hccl_ 成员** | 声明但**不调用** | 声明但**不调用** | 声明且调用 (`InitV2`, `SetCcTilingV2`, `GetWindowsInAddr`, `Finalize`) | 声明且调用 |
| **4. Window 地址获取** | 直接字段 `localWindowsIn` / `remoteRes[].nextDevicePtr` | `GetBaseWindAddrByRankId()` 辅助函数 | `hccl_.GetWindowsInAddr(rankId)` | `hccl_.GetWindowsInAddr/OutAddr(rankId)` |
| **5. winDataOffset** | `dataState * (totalWinSize / 2) + maxBS * topK * hAlign` | `dataState * (baseWindSize / 2) + maxBS * topK * hAlign` | `dataState * ((totalWinSize - STATE_SIZE*4) / 2) + min(maxBS, perRoundTokens) * topK * hAlign` | `bufferId_ * halfWinSize` (双buffer偏移) |
| **6. 状态地址公式** | `localWindowsExp + dataState*WIN_STATE_OFFSET` | `GetStatusDataSpaceGm() + dataState*WIN_STATE_OFFSET` | `GetWindowsInAddr(rank) + totalWinSize - STATE_SIZE*2 + dataState*WIN_STATE_OFFSET` | `GetWindowsInAddr(rank) + totalWinSize_ - STATE_SIZE*3` |
| **7. 数据发送** | `DataCopyPad` 写共享窗口 GM | `DataCopyPad` 写共享窗口 GM | `DataCopyPad` 写自身窗口 GM (HCCL底层完成跨节点) | 机内: `DataCopy` 写 IPC 共享内存; 机间: `hccl_.BatchWrite<true>()` (RDMA) |
| **8. Token 结构** | token + expandIdx(3个int32: rankId, tokenIdx, k) | token + expandIdx + 可选 MX scale | token + expandIdx(3个int32) + 可选 dynamic scale (同 A3) | token + expertIds(kAlign) + weights(kAlign) + tokenIdx + scales |
| **9. 接收等待** | 轮询 `DataCopy` + `ReduceSum` 检查 sum == statusNumPerCore | 同 A3 | 同 A3 (轮询 `DataCopy` + `ReduceSum`, float 1.0 求和) | RDMA: 轮询 `DataCopy` 检查 `FLAG_VALUE(0xFFFFFFFF)`; IPC: 轮询 magic flag (`MergeMagicWithValue`) |
| **10. Round 等待** | `SetRoundStatus` + `WaitRoundStatus` (Sum 轮询, blockIdx==0) | 同 A3 | 同 A3 (`SetRoundStatus` + `WaitRoundStatus`, blockIdx==0) | IPC magic 两步握手 (step1=数据就绪, step2=清理完成) |
| **11. 状态标志大小** | `STATE_OFFSET=32B` | `STATE_OFFSET=32B` | `STATE_OFFSET=32B` | `STATE_OFFSET=512B` |
| **12. 状态窗口偏移** | `STATE_WIN_OFFSET=950KB` | `STATE_WIN_OFFSET=1050KB` | `STATE_WIN_OFFSET=950KB` | `SELF_STATE_OFFSET=512KB` |
| **13. WIN_STATE_OFFSET** | `500KB` | `550KB` | `500KB` | N/A |
| **14. COMBINE_STATE_WIN_OFFSET** | `4MB` | `4MB` | **`8MB`** | N/A (不同布局) |
| **15. Double buffer** | `dataState` 翻转 (0/1) | `dataState` 翻转 (0/1) | `dataState` 翻转 (0/1) (读窗口状态区, 翻转写回) | `bufferId_` 翻转 (通过 `bufferChosenGlobal_`) |
| **16. Magic 机制** | 无 (float 1.0 状态值) | 无 (float 1.0 状态值) | 无 (float 1.0 状态值, 无 magic) | 有 (`MergeMagicWithValue(magic, val) = magic*2+val`) |
| **17. 超时检测** | 有 (`TimeOutDetection`) | 无 | 无 (仅诊断时间统计, 无超时退出) | 无 |
| **18. 量化支持** | DynamicQuant (int8) | DynamicQuant + IsMxQuant (int8/fp8/fp4) | 仅 DynamicQuant (int8, 无 StaticQuant/MX) | StaticQuant + DynamicQuant (int8) |
| **19. TP 域支持** | 是 (`winContext_[COMM_TP_IDX]`) | 是 | 否 (声明 `COMM_TP_IDX` 但未赋值/使用) | 否 (仅 EP) |
| **20. Server 概念** | 无 (扁平 rank) | 无 (扁平 rank) | 无 (扁平 rank, 无 `SERVER_RANK_SIZE`) | 有 (`SERVER_RANK_SIZE=8`) |
| **21. HCCL 生命周期** | 无 (仅用窗口) | 无 (仅用窗口) | `InitV2` -> ... -> `Finalize` | `InitV2` -> ... -> `Finalize` |
| **22. 核间分工** | blockIdx 分配 expert/token | 同 A3 | blockIdx 分配 expert/token (同 A3) | aivId 分配 token/server, aivId==0 负责 BatchWrite |
| **23. 多轮支持** | 有 (`round`, `perRoundTokens`, `realRound`) | 同 A3 | 有 (`round`, `perRoundTokens`, `realRound`, 同 A3) | 无 (单轮) |
| **24. Process 步骤** | while(round): InputToShare -> SetStatus -> WaitStatus -> ShareToOutput -> SetRoundStatus -> WaitRoundStatus | 同 A3 | while(round): InputToShare -> SetStatus -> WaitStatus -> ShareToOutputLongSeq -> SetRoundStatus -> WaitRoundStatus -> Finalize | 16 步: Input2Win -> WriteRdmaCntInfo -> DispatchBetweenServer -> WaitWindow -> SetIpcFlag(step1) -> WaitIpcFlag(step1) -> Ipc2Out -> Cleanup -> SetIpcFlag(step2) -> WaitIpcFlag(step2) -> Finalize |
| **25. 关键常量** | `WIN_STATE_OFFSET=500KB`, `COMBINE_STATE_WIN_OFFSET=4MB`, `NOTIFY_DISPATCH_BUFF_OFFSET=102MB` | `WIN_STATE_OFFSET=550KB`, `A5_MTE_STATE_WIN_SIZE=4MB` | `WIN_STATE_OFFSET=500KB`, `COMBINE_STATE_WIN_OFFSET=8MB`, `NOTIFY_DISPATCH_BUFF_OFFSET=202MB`, `STATE_SIZE=2MB` | `RDMA_DATA_SIZE=800MB`, `IPC_DATA_OFFSET=4MB`, `SERVER_RANK_SIZE=8` |
| **26. winContext_ 使用** | 是 (直接字段访问) | 是 (helper 函数访问) | 否 (赋值但从未使用, 全部通过 `hccl_.GetWindowsInAddr()`) | 是 (用于 `winSize` 字段) |

### 5.4 NormalCombine (CamMoeCombineNormal)

| 步骤 | A3 (`cam_moe_combine_normal.h`) | A5 (`cam_moe_combine_normal_a5.h`) | A2 Base (`ops2/cam_moe_combine_normal.h`) | A2 Layered (`moe_distribute_combine_a2_layered.h`) |
|------|--------------------------------|-----------------------------------|------------------------------------------|---------------------------------------------------|
| **1. 通信上下文** | `HcclOpResParam*` | `HcclOpParam*` | `Hccl<AICPU>` + `HcclOpResParam*` | `Hccl<AICPU>` + `HcclOpResParam*` |
| **2. 上下文初始化** | 直接 cast `GetHcclContext<0>()` | 同 A3 | `hccl_.InitV2()` + `SetCcTilingV2()` | `hccl_.InitV2()` + `SetCcTilingV2()` |
| **3. Window 地址获取** | 直接字段 `remoteRes[].nextDevicePtr->windowsIn` | `GetBaseWindAddrByRankId()` (直接索引 `windowsIn[rankId]`) | `hccl_.GetWindowsInAddr(rankId)` | `hccl_.GetWindowsInAddr(rankId)` + IPC 地址表 |
| **4. Magic/双缓冲** | magic (0/1) 读写 `localWindowsExp + MAGIC_WIN_OFFSET` | magic (0/1) 读写 `GetStatusDataSpaceGm() + MAGIC_WIN_OFFSET` | magic (0/1) 读写 `hccl_.GetWindowsInAddr() + totalWinSize_ - STATE_SIZE + MAGIC_WIN_OFFSET` | `bufferId_` XOR 翻转 + `stateValue_` (0/1) |
| **5. winDataOffset** | `magic * (totalWinSize / 2)` | `magic * (baseWindSize / 2)` | `magic * ((totalWinSize_ - 4*STATE_SIZE) / 2)` | `bufferId_ * halfWinSize_` |
| **6. 数据发送** | `DataCopyPad` 写远端 window (GM->UB->GM) | 同 A3 | 同 A3 | IPC: `DataCopy` 写共享内存; RDMA: `hccl_.BatchWrite<true>()` |
| **7. 数据聚合** | `ReadBufferAndWeightedSum`: Cast->Muls(weight)->Add | 同 A3 | 同 A3 | `SumToWindow`: IPC 读取->Cast->Muls->Add; `SumToServer`: RDMA window 读取->Cast->Add |
| **8. 接收等待** | 轮询 `DataCopy` + `Sum` 检查 `current == target` | 同 A3 | 同 A3 | `WaitBuffCopy` 轮询 flag; `WaitIPC` 轮询 `GM2IPC_SYNC_FLAG+magic`; `WaitDispatch` 轮询 `sumOfFlag == sumTarget_` |
| **9. 跨核同步** | `SyncFunc` + `PipeBarrier` (无 `SyncAll`) | 同 A3 (无 `SyncAll`) | 同 A3 (无 `SyncAll`) | `SyncAll<true>()` (多次) + `SyncFunc` + `PipeBarrier` |
| **10. hccl_.Finalize** | 否 | 否 | 是 | 是 |
| **11. 发送核分配** | 按 token 均分 | rank-major 两级分区 (A>=W 按 rank 分核组, A<W 退化) | 按 token 均分 | 按 rank 分核 + server 分层 |
| **12. Process 步骤** | 2 步: CopyBufferToShareAndSetStatus -> ReadBufferFromRemote | 同 A3 (2步) | 同 A3 (2步 + Finalize) | 7 步: AlltoAllDispatch(IPC) -> SumToWindow -> AlltoAllServerDispatch(RDMA) -> SetStatus -> Preload -> WaitDispatch -> SumToServer -> Finalize |
| **13. COMBINE_STATE_WIN_OFFSET** | 4MB | 4MB | **8MB** | N/A (使用不同布局) |
| **14. MAGIC_WIN_OFFSET** | 975KB | **1100KB** | 975KB | N/A (使用 `MAGIC_OFFSET`) |
| **15. BATCH_SRC_INFO_CNT** | 无 | 128 (批量读 srcInfo) | 无 | N/A |
| **16. Token 结构** | tokenSrcInfo: 3个uint32 (rankId, tokenId, topkId) | 同 A3 | 同 A3 | expandX + expandIdx + offsetInner/Outer + countOuter |

### 5.5 DispatchV2 (MoeDistributeDispatchV2 低延迟)

| 步骤 | A3/Base (`moe_distribute_dispatch_v2.h`) | A5 (`moe_distribute_dispatch_v2_a5.h`) | CCU (`moe_distribute_dispatch_v2_ccu.h`) | A2 Single (`ops2/moe_distribute_dispatch_v2_single.h`) | A2 Layered (`ops2/moe_distribute_dispatch_v2_layered.h`) |
|------|------------------------------------------|----------------------------------------|------------------------------------------|--------------------------------------------------------|---------------------------------------------------------|
| **1. 通信上下文** | `HcclOpParam*` (EP+TP) | `HcclOpParam*` (EP+TP) | `HcclCombineOpParam*` | `HcclOpResParam*` (EP+TP) | `HcclA2CombineOpParam*` |
| **2. HCCL 对象** | 无 | 无 | `Hccl<CCU>` | `Hccl<AICPU>` | `Hccl<AICPU>` |
| **3. 上下文初始化** | 直接 cast | 直接 cast | `hccl_.InitV2()` + `SetCcTilingV2()` | `hccl_.InitV2()` + `SetCcTilingV2()` + cast | `hccl_.InitV2()` + `SetCcTilingV2()` + cast |
| **4. dataState 初始化** | `InitWinState()` helper | `InitWinState()` helper | 双 buffer 读 `statusGT(0)` 翻转 | 直接读写 `hccl_.GetWindowsInAddr() + STATE_WIN_OFFSET` + `DataCacheCleanAndInvalid` | `bufferId_` 从 `bufferIdGlobal_` 读取 |
| **5. Window IN 获取** | `GetBaseWindAddrByRankId(ctx, rankId, curRankId)` | 同 A3 | `context->windowsOut[0]` | `hccl_.GetWindowsInAddr(rankId) + TOTAL_STATE_OFFSET` | `hccl_.GetWindowsInAddr(rankId_)` + `halfWinSize_ * bufferId_` |
| **6. Window OUT 获取** | `GetBaseWindStateAddrByRankId()` | 同 A3 | `context->windowsOut[0]` (sendBuf=workspace) | N/A (同 IN) | `hccl_.GetWindowsOutAddr(rankId_)` + `halfWinSize_ * bufferId_` |
| **7. winDataSizeOffset** | `dataState * (totalWinSize / 2)` | `dataState * (baseWindSize / 2)` | N/A (perRankDataSize 计算) | `dataState * (totalWinSize / 2)` | `bufferId_ * halfWinSize_` |
| **8. 数据发送** | `DataCopyPad` 写窗口 | `DataCopyPad` (封装为 `ProcessToken()`) | `hccl_.AlltoAllvWrite<true>(sendBuf, sendOffset, sendSize, recvOffset, localDataSize)` | `DataCopyPad` 写窗口 | `AIVRDMAPostSend()` (直接操作 RoCE 硬件队列: 构造 WQE -> cacheWriteThrough -> 写 DoorBell) |
| **9. 状态写入** | `DataCopy<int32_t>` 写 1.0 标记 | 同 A3 | N/A (HCCL 框架处理) | `DataCopy<int32_t>` 写 1.0 标记 | `AIVRDMAPostSend` 发送 flag (32B) |
| **10. 接收等待** | `DataCopy` + `ReduceSum` 轮询 `sumOfFlag != compareTarget` | 同 A3 (非阻塞) | `hccl_.Wait(hcclHandleId_)` | `DataCopy` + `ReduceSum` (无超时检测) | Token flag 轮询: 检查 `SHOULD_SEND_FLAG_VALUE(0x0f0f0f0f)` / `END_OF_WRITE_FLAG_VALUE(0xffffffff)` + IPC flag 轮询 |
| **11. 超时检测** | 有 (`TimeOutDetection`, 50000us) | 无 (while 中不调用) | 无 | 无 | 无 |
| **12. 核间同步** | `SyncAll` + `SyncCntOnCore` | `SyncAll` + `SyncCntOnCore` | `SyncAll` (3次) | `SyncAll` + `SyncCntOnCore` | `SyncAll` + IPC flag (SetIpcFlag/WaitIpcFlag) |
| **13. TP 域支持** | 是 (`IsNeedAllgather`) | 是 | 否 | 是 (`IsNeedAllgater`) | 否 |
| **14. 量化支持** | Static/Dynamic | Static/Dynamic/MX/FP4/FP8 | Static/Dynamic/MX/PerGroup (QuantMode 枚举) | Static/Dynamic | Static/Dynamic |
| **15. STATE_WIN_OFFSET** | 950KB | **1050KB** | N/A | 950KB | N/A |
| **16. 额外状态偏移** | 无 | `A5_MTE_STATE_WIN_SIZE` (4MB) | `COUNT_OFFSET=512`, `STATUS_OFFSET=512` | `TOTAL_STATE_OFFSET=3MB` | `IPC_*` 系列偏移 |
| **17. Process 流程** | AlltoAllDispatch -> SetStatus -> WaitDispatch -> LocalWindowCopy -> [AllGather] -> UpdateTokenNums | 同 A3 | CalcTokenActiveMask -> TokenScatter(Sort+Scatter) -> Communication(AlltoAllvWrite+Wait) -> TokenGather -> Finalize | 同 A3 + Finalize | ReorderTokens -> SendDataToServer(RDMA) + Win2Ipc -> SetIpcFlag -> WaitIpcFlag -> Ipc2Out -> CleanUp -> Finalize |
| **18. hccl_.Finalize** | 否 | 否 | 是 (aivId==0) | 是 | 是 |
| **19. QP/RDMA 信息** | N/A | N/A | N/A | N/A | `qp_info_ = winContext_->aiRMAInfo` (用于 `AIVRDMAPostSend`) |
| **20. 双缓冲切换** | `dataState_` 翻转 | `dataState_` 翻转 | `statusGT(0)` 翻转 | `dataState_` 翻转 | `bufferId_` 翻转 (`bufferChosenGlobal_`) |

### 5.6 CombineV2 (MoeDistributeCombineV2 低延迟)

| 步骤 | A3/Base (`moe_distribute_combine_v2.h`) | A5 (`moe_distribute_combine_v2_a5.h`) | CCU (`moe_distribute_combine_v2_ccu.h`) | A2 Single (`ops2/moe_distribute_combine_v2_single.h`) | A2 Layered (`ops2/moe_distribute_combine_v2_layered.h`) |
|------|------------------------------------------|----------------------------------------|------------------------------------------|--------------------------------------------------------|---------------------------------------------------------|
| **1. 通信上下文** | `HcclOpParam*` (EP+TP) | `HcclOpParam*` (EP+TP) | `HcclCombineOpParam*` | `HcclOpResParam*` (EP+TP) | `HcclA2CombineOpParam*` |
| **2. HCCL 对象** | 无 | 无 | `Hccl<CCU>` | `Hccl<AICPU>` | `Hccl<AICPU>` |
| **3. 上下文初始化** | 直接 cast | 直接 cast | `hccl_.InitV2()` + `SetCcTilingV2()` | `hccl_.InitV2()` + `SetCcTilingV2()` + cast | `hccl_.InitV2()` + `SetCcTilingV2()` + cast |
| **4. Window IN 获取** | `GetBaseWindAddrByRankId(ctx, rankId, curRankId)` | 同 A3 (A5底层: `windowsIn[rankId]+A5_MTE_STATE_WIN_SIZE`) | `context->windowsOut[0]` + 双buffer偏移 | `hccl_.GetWindowsInAddr(rankId) + TOTAL_STATE_OFFSET` (忽略 domain 参数) | `hccl_.GetWindowsInAddr(rankId_)` + `halfWinSize_ * bufferId_` |
| **5. Window State 获取** | `GetBaseWindStateAddrByRankId()` | 同 A3 (A5底层: `windowsIn[rankId]`) | N/A (HCCL 框架处理) | `hccl_.GetWindowsInAddr(rankId) + winDataSizeOffset_ + winStatusOffset_` (用 IN 地址) | `windowInGM_ + dataSpaceSize_` (状态区在 window 尾部) |
| **6. winDataSizeOffset** | `dataState * (totalWinSize / 2)` | `dataState * (baseWindSize / 2)` | N/A (perRankDataSize 计算) | `dataState * (totalWinSize / 2)` | `bufferId_ * halfWinSize_` |
| **7. 数据发送** | `DataCopyPad` 写远端窗口 | 同 A3 | `hccl_.AlltoAllvWrite<true>()` | `DataCopyPad` 写远端窗口 | IPC: `DataCopy` 写共享内存; RDMA: `AIVRDMAPostSend()` |
| **8. 数据聚合/规约** | `CustomAdd` (Cast->Add->Cast) 或 `Int8QuantProcess`; 最终: Cast->Muls->Add | 同 A3 + `ProcessExpert` 抽取 | `CalculateMoeResult`: Cast->Muls->Add | `CustomAdd` + `Int8QuantProcess`; 最终: Cast->Muls->Add | `SumToWindow`: Cast->Muls/Axpy->Add + 可选量化; `SumToServer`: Cast->Add + 可选反量化 |
| **9. 接收等待** | `DataCopy`+`Sum` 轮询 `localState` vs target (阻塞式 while) | `DataCopy`+`Sum` 非阻塞返回 bool + `tokenStatusTensor` 跟踪 | `hccl_.Wait(hcclHandleId_)` | `DataCopy`+`Sum` 轮询 (无超时检测) | `WaitIPC`: 轮询 `GM2IPC_SYNC_FLAG+magic`; `WaitDispatch`: 轮询 `sumOfFlag == sumTarget_` |
| **10. 超时检测** | 有 (50000us) | 无 | 无 | 无 | 无 |
| **11. 核间同步** | `PipeBarrier` + `SyncFunc` | 同 A3 + `PipeBarrier<PIPE_ALL>` 新增 | `SyncAll` (3次) | `SyncAll` + `SyncFunc` | `SyncAll` (多次) + `SyncFunc` |
| **12. TP 域支持** | 是 (`IsNeedReduceScatter`) | 是 | 否 | 是 (`IsNeedReduceScatter`) | 否 |
| **13. 量化支持** | Int8Quant | Int8Quant | 不支持 | Int8Quant | DynamicQuant (`ExpandXTransType`) |
| **14. 特殊专家** | 支持 (zero/copy/const) | 支持 | 不支持 | 不支持 | 不支持 |
| **15. Magic 防伪** | 无 | 无 | 无 | 无 | 有 (`magicValue` + flag: `GM2IPC_SYNC_FLAG=12345`, `RDMA_TOKEN_ARRIVED_FLAG=123`, `RDMA_TOKEN_END_FLAG=321`) |
| **16. 双缓冲** | `dataState_` 翻转 | `dataState_` 翻转 | `statusGT(0)` 翻转 | `dataState_` 翻转 | `bufferId_` XOR 翻转 + `sumTarget_` 翻转 |
| **17. STATE_WIN_OFFSET** | 975KB | **1100KB** | N/A | 975KB | N/A |
| **18. STATE_CHECK_OFFSET** | 1000KB | **1150KB** | N/A | N/A | N/A |
| **19. Process 流程** | ReduceScatterTrans -> BuffInit -> SetWaitTpStatusAndDisPatch -> AlltoAllBuffInit -> LocalWindowCopy | 同 A3 + PipeBarrier 新增 | PrepareSendData -> Communication(AlltoAllvWrite+Wait) -> Calculate -> Finalize | 同 A3 + SyncAll + Finalize | GM2IPC -> WaitIPC -> SumToWindow -> AlltoAllServerDispatch(RDMA) -> WaitDispatch -> SumToServer -> Finalize |
| **20. hccl_.Finalize** | 否 | 否 | 是 (aivId==0) | 是 | 是 |
| **21. QP/RDMA 信息** | N/A | N/A | N/A | N/A | `qp_info_ = winContext_->aiRMAInfo` |

### 5.7 aclnn 算子调用适配

`deep_ep.cpp` 中的适配方案：

```cpp
// 当前代码: 硬编码机型分支
if (soc_version == op::SocVersion::ASCEND910B) {
    EXEC_NPU_CMD(aclnnNotifyDispatchA2, ...);
} else {
    EXEC_NPU_CMD(aclnnNotifyDispatch, ...);
}

// 适配后: 通过 MachineType 选择算子
auto type = CommFactory::detect_machine_type(soc_version_str);
switch (type) {
    case MachineType::A2:
        EXEC_NPU_CMD(aclnnNotifyDispatchA2, ...);
        break;
    default:
        EXEC_NPU_CMD(aclnnNotifyDispatch, ...);
}
```

完整算子映射表：

| 功能 | A2 | A3 | A5 | CCU |
|------|-----|-----|-----|-----|
| Layout | `aclnnDispatchLayout` | `aclnnDispatchLayout` | `aclnnDispatchLayout` | `aclnnDispatchLayout` |
| Notify | `aclnnNotifyDispatchA2` | `aclnnNotifyDispatch` | `aclnnNotifyDispatch` | `aclnnNotifyDispatch` |
| Intranode Dispatch | N/A | `aclnnCamMoeDispatchNormal` | `aclnnCamMoeDispatchNormal` | N/A |
| Internode Dispatch | `aclnnDispatchNormalA2` | N/A | N/A | N/A |
| Low-latency Dispatch | `aclnnMoeLowLatencyDispatchV2` (comm_alg="fullmesh") | `aclnnMoeLowLatencyDispatchV2` (comm_alg="fullmesh_v1") | `aclnnMoeLowLatencyDispatchV2` (comm_alg="fullmesh_v1") | `aclnnMoeLowLatencyDispatchV2` (comm_alg="ccu") |
| Intranode Combine | N/A | `aclnnCamMoeCombineNormal` | `aclnnCamMoeCombineNormal` | N/A |
| Internode Combine | `aclnnMoeDistributeCombineA2` | N/A | N/A | N/A |
| Low-latency Combine | `aclnnMoeLowLatencyCombineV2` (comm_alg="fullmesh") | `aclnnMoeLowLatencyCombineV2` (comm_alg="fullmesh_v1") | `aclnnMoeLowLatencyCombineV2` (comm_alg="fullmesh_v1") | `aclnnMoeLowLatencyCombineV2` (comm_alg="ccu") |
| FusedDeepMoe | N/A | `aclnnFusedDeepMoe` (INT8) | `aclnnFusedDeepMoe` (FP8/FP4/MX) | N/A |

---

## 6. 落地实施步骤

### 步骤 1: 创建统一通信接口头文件

创建 `csrc/deepep/comm/` 目录，放入以下文件：

```
csrc/deepep/comm/
├── comm_types.hpp              # 机型/通信模式/量化类型枚举
├── unified_comm_context.hpp    # 统一接口定义 (纯虚类)
├── comm_factory.hpp            # 工厂模式创建通信实例
├── window_comm_impl.hpp        # A3/A5 窗口通信实现
├── ccu_comm_impl.hpp           # CCU 高层 API 通信实现
├── batch_write_comm_impl.hpp   # A2 Single BatchWrite 通信实现
└── layered_comm_impl.hpp       # A2 Layered 分层通信实现
```

**风险**：低。纯新增文件，不影响现有代码。

### 步骤 2: 改造 deep_ep.cpp 中的机型判断

将 `deep_ep.cpp` 中分散的 `soc_version` 判断和 `__DAV_C310__` 宏统一为 `MachineType` 枚举：

```cpp
// deep_ep.cpp 构造函数中
auto soc_version_str = get_soc_version_string();
machine_type_ = CommFactory::detect_machine_type(soc_version_str);
is_layered_ = CommFactory::detect_is_layered(machine_type_);
is_ccu_ = CommFactory::detect_is_ccu();
comm_alg_ = CommFactory::get_comm_alg(machine_type_, is_ccu_);
```

**风险**：低。仅替换判断逻辑，不改变调用路径。

### 步骤 3: 改造 Kernel 中的通信上下文获取

逐个 Kernel 文件替换通信上下文类型为 `UnifiedCommContext`：

**优先级**：
1. `notify_dispatch*.h` (3 个文件) — 最独立，改动风险最低
2. `dispatch_layout*.h` (3 个文件) — 纯计算，无通信接口
3. `moe_distribute_dispatch_v2*.h` (5 个文件) — 核心算子
4. `moe_distribute_combine_v2*.h` (5 个文件) — 核心算子
5. `cam_moe_*_normal*.h` (4 个文件) — 依赖前两者

**风险**：中。需要确保各机型的窗口地址计算与原始实现完全一致。

### 步骤 4: 合并 ops/ 和 ops2/ 中的重复算子

消除以下重复文件对：

| ops/ 文件 | ops2/ 文件 | 合并策略 |
|-----------|-----------|---------|
| `dispatch_layout.h` | `dispatch_layout.h` | 保留 A3 版本，A2 分支通过 TilingKey 选择 |
| `dispatch_layout_a2.h` | `dispatch_layout_a2.h` | 合并为一个文件，ops2 版本为准 (含 tokenIdx 输出) |
| `notify_dispatch.h` | `notify_dispatch.h` | 保留一份，通过 `#ifdef` 区分 |
| `cam_moe_dispatch_normal.h` | `cam_moe_dispatch_normal.h` | 同上 |
| `cam_moe_combine_normal.h` | `cam_moe_combine_normal.h` | 同上 |

**风险**：中。需要仔细对比两个版本的差异。

### 步骤 5: 统一 CMakeLists.txt

将 `ops/` 和 `ops2/` 的 CMakeLists.txt 合并：

```cmake
# 统一源文件列表
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/op_host ops_srcs)

# A5 编译时替换 FusedDeepMoe 源文件 (保持现有机制)
if("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950")
    list(REMOVE_ITEM ops_srcs ... fused_deep_moe.cpp ...)
    list(APPEND ops_srcs ... fused_deep_moe_a5/fused_deep_moe.cpp ...)
    list(APPEND op_codegen_options -D__DAV_C310__)
endif()

# 添加统一通信抽象层源文件
list(APPEND ops_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/../comm/comm_factory.cpp)
```

**风险**：低。CMake 配置变更，可通过编译验证。

---

## 附录 A: 各机型通信接口完整清单

### A.1 所有通信接口方法

| # | 接口方法 | A3 | A5 | CCU | A2 Single | A2 Layered | 说明 |
|---|---------|-----|-----|-----|-----------|------------|------|
| 1 | `get_window_in_addr(ctx, rank)` | 直接字段 | helper函数 | N/A | Hccl方法 | Hccl方法 | 获取输入窗口地址 |
| 2 | `get_window_out_addr(ctx, rank)` | 直接字段 | helper函数 | N/A | Hccl方法 | Hccl方法 | 获取输出窗口地址 |
| 3 | `get_window_data_size(ctx)` | `winSize` | `GetWinSize()-state` | 0 | `winSize` | `winSize-IPC*2` | 窗口数据区大小 |
| 4 | `get_status_addr(ctx, rank)` | `winExp+offset` | `GetStatusGm()+offset` | `winIn+offset` | `winIn+winSize-state*3` | `winIn+winSize-IPC*2` | 状态区地址 |
| 5 | `get_magic_value(ctx)` | `*winExp` | `*GetStatusGm()` | `*winIn+offset` | `*winIn+500KB` | `*winIn+IPC_SIZE` | 通信轮次 magic |
| 6 | `send_data(ctx, local, remote, size, rank)` | DataCopyPad | DataCopyPad | N/A | BatchWrite | RDMA/IPC | 单条发送 |
| 7 | `batch_write(ctx, items, count)` | N/A | N/A | N/A | `hccl_.BatchWrite` | `hccl_.BatchWrite` | 批量发送 |
| 8 | `allto_all_write(send, counts, ...)` | N/A | N/A | `AlltoAllvWrite` | N/A | N/A | 全互联写入 (CCU专用) |
| 9 | `wait_recv(ctx, rank, magic)` | magic掩码轮询 | magic掩码非阻塞 | status轮询 | status轮询 | CQ+IPC flag轮询 | 等待接收完成 |
| 10 | `get_ipc_info()` | N/A | N/A | N/A | N/A | `IpcShareInfo` | 获取IPC共享内存信息 |
| 11 | `set_ipc_flag(rank, magic)` | N/A | N/A | N/A | N/A | 写flag+cache flush | 设置IPC完成标志 |
| 12 | `wait_ipc_flag(rank, magic)` | N/A | N/A | N/A | N/A | 轮询flag | 等待IPC完成 |
| 13 | `get_rdma_info()` | N/A | N/A | N/A | N/A | `RdmaHwInfo` | 获取RDMA硬件信息 |
| 14 | `rdma_post_send(rank, local, remote, size)` | N/A | N/A | N/A | N/A | 构造WQE+SGE+doorbell | RDMA直接发送 |
| 15 | `sync_all()` | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` | `SyncAll<true>()` | 核间同步 |
| 16 | `is_need_allgather()` | false | false | false | true | false | 是否需要TP AllGather |
| 17 | `is_need_reduce_scatter()` | false | false | false | true | false | 是否需要TP ReduceScatter |
| 18 | `server_num()` | 1 | 1 | 1 | 1 | `rankSize/8` | server数量 |

### A.2 窗口地址获取方式对比

```
A3 (HcclOpResParam):
  本地: ctx->localWindowsIn + rankId * offset
  远端: ((HcclRankRelationResV2*)ctx->remoteRes[rankId].nextDevicePtr)->windowsIn

A5 (HcclOpParam):
  本地/远端: GetBaseWindStateAddrByRankId(winContext_, rankId, curRankId)
  状态区: GetStatusDataSpaceGm(winContext_) + STATE_WIN_OFFSET
  窗口大小: GetWinSize(winContext_) - A5_MTE_STATE_WIN_SIZE

CCU (HcclCombineOpParam):
  不直接访问窗口地址，通过 AlltoAllvWrite API 一步完成

A2 Single (HcclOpResParam + Hccl<>):
  本地/远端: hccl_.GetWindowsInAddr(rankId)
  状态区: hccl_.GetWindowsInAddr(rankId) + winSize - STATE_SIZE * 3

A2 Layered (HcclOpResParam + Hccl<> + aiRMAInfo):
  窗口: hccl_.GetWindowsInAddr(rankId)
  状态区: hccl_.GetWindowsInAddr(rankId) + winSize - IPC_BUFF_MAX_SIZE * 2
  IPC: ctx->zeroCopyIpcPtrs[localRank]
  RDMA: ((HcclAiRMAInfo*)a2Param->aiRMAInfo)->sqPtr
```

### A.3 状态同步机制对比

```
A3:
  状态地址: winContext_->localWindowsExp + 900KB
  同步方式: (*status & MAGIC_MASK) == (magic & MAGIC_MASK)
  阻塞模式: while 轮询

A5:
  状态地址: GetStatusDataSpaceGm(winContext_) + 1000KB
  同步方式: 同A3 (magic掩码比较)
  非阻塞模式: 检查一次后返回 bool

CCU:
  状态地址: winIn + 950KB
  同步方式: *status == expectedValue (直接比较)

A2 Single:
  状态地址: winIn + winSize - 3 * 2MB
  同步方式: *status == magic (直接比较)

A2 Layered:
  跨server: 轮询 RDMA CQ (Completion Queue)
  同server: 轮询 IPC flag (*flagAddr == magic)
  flag地址: shareAddrs[rank%8] + 2MB (IPC_DATA_OFFSET)
```

### A.4 环境变量控制矩阵

| 环境变量 | 机型 | 默认值 | 作用 |
|---------|------|--------|------|
| `HCCL_INTRA_PCIE_ENABLE` | A2 | 未设置 | 值为"1"时启用 PCIe 层内通信 |
| `HCCL_INTRA_ROCE_ENABLE` | A2 | 未设置 | 值为"0"时禁用 RoCE 层内通信。两者同时满足时启用 layered 模式 |
| `MOE_ENABLE_CCU` | A3/A5 | 0 | 值为1时启用 CCU 通信算法 |
| `MOE_SHARED_EXPERT_RANK_NUM` | 全部 | 0 | 共享专家 rank 数量 |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND` | 全部 | 1 | dispatch 多轮处理的轮数 [1, 256] |
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | 全部 | 8192 | 每轮处理的 token 数 [32, 8192] |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ` | 全部 | 0 | 是否为 combine 启用多轮 |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | 全部 | 1 | expert_token_nums 输出类型: 0=前缀和, 1=实际数量 |
| `MOE_ENABLE_TOPK_NEG_ONE` | 全部 | 0 | 是否启用 topk_idx 中 -1 值支持 (生成 active_mask) |

### A.5 TilingKey 体系

```
基数:
  A2  = 20000
  A3  = 30000  (INIT_TILINGKEY)
  A5  = 50000
  CCU = 60000

附加位:
  + SCALES(10)           // 有 scales 输入
  + TP_WORLD_SIZE(100)   // TP=2
  + COMM_ALG(1000)       // commAlg=fullmesh_v2

示例:
  A3 无量化无TP:           30000
  A3 有量化:               30010
  A3 有量化+TP=2:          30110
  A5 无量化:               50000
  CCU 无量化:              60000
  A2 无量化:               20000
```

### A.6 量化类型支持矩阵

| 量化类型 | A2 | A3 | A5 | CCU | quant_mode 值 |
|---------|-----|-----|-----|-----|--------------|
| NONE (BF16/FP16) | YES | YES | YES | YES | 0 |
| INT8 静态 | YES | NO | NO | NO | 1 |
| INT8 动态 | YES | YES | NO | NO | 2 |
| FP8 E5M2 | NO | NO | YES | NO | 3 |
| FP8 E4M3 | NO | NO | YES | NO | 3 |
| FP4 E2M1 | NO | NO | YES | NO | 4 |
| MX FP8 (e8m0 scale) | NO | NO | YES | NO | 4 (dispatch), 3 (fused) |
| MX FP4 (e8m0 scale) | NO | NO | YES | NO | 4 |
| Per-token FP8 | NO | NO | YES | NO | 5 |

> 注: A5 的 FP8/FP4 量化需要 `__DAV_C310__` 编译宏，仅在 `SOC_VERSION=Ascend950` 时可用。

---

## 附录 B: 文件变更清单

### B.1 新增文件

| 文件路径 | 说明 |
|---------|------|
| `csrc/deepep/comm/comm_types.hpp` | 机型/通信模式/量化类型枚举定义 |
| `csrc/deepep/comm/unified_comm_context.hpp` | 统一通信上下文纯虚接口 |
| `csrc/deepep/comm/comm_factory.hpp` | 通信实例工厂 |
| `csrc/deepep/comm/window_comm_impl.hpp` | A3/A5 窗口通信实现 |
| `csrc/deepep/comm/ccu_comm_impl.hpp` | CCU 高层 API 通信实现 |
| `csrc/deepep/comm/batch_write_comm_impl.hpp` | A2 Single BatchWrite 通信实现 |
| `csrc/deepep/comm/layered_comm_impl.hpp` | A2 Layered 分层通信实现 |

### B.2 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `csrc/deepep/deep_ep.cpp` | 统一机型判断为 `MachineType` 枚举，消除分散的 `soc_version` 检查 |
| `csrc/deepep/deep_ep.hpp` | 添加 `machine_type_` 等成员变量 |
| `csrc/deepep/config.hpp` | 添加 `MachineType` 相关声明 |
| `csrc/deepep/CMakeLists.txt` | 添加 `comm/` 目录源文件 |
| `csrc/deepep/ops/op_kernel/notify_dispatch.h` | 替换 `HcclOpResParam*` 为 `UnifiedCommContext*` |
| `csrc/deepep/ops/op_kernel/notify_dispatch_a5.h` | 替换 `HcclOpParam*` 为 `UnifiedCommContext*` |
| `csrc/deepep/ops2/op_kernel/notify_dispatch_a2.h` | 替换 `HcclOpResParam*` + `Hccl<>` 为 `UnifiedCommContext*` |
| `csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2.h` | 替换通信上下文 |
| `csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2_a5.h` | 替换通信上下文 |
| `csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2_ccu.h` | 替换通信上下文 |
| `csrc/deepep/ops2/op_kernel/moe_distribute_dispatch_v2_single.h` | 替换通信上下文 |
| `csrc/deepep/ops2/op_kernel/moe_distribute_dispatch_v2_layered.h` | 替换通信上下文 |
| `csrc/deepep/ops/op_kernel/moe_distribute_combine_v2.h` | 替换通信上下文 |
| `csrc/deepep/ops/op_kernel/moe_distribute_combine_v2_a5.h` | 替换通信上下文 |
| `csrc/deepep/ops/op_kernel/moe_distribute_combine_v2_ccu.h` | 替换通信上下文 |
| `csrc/deepep/ops2/op_kernel/moe_distribute_combine_v2_single.h` | 替换通信上下文 |
| `csrc/deepep/ops2/op_kernel/moe_distribute_combine_a2_layered.h` | 替换通信上下文 |

### B.3 可合并的重复文件

| ops/ 文件 | ops2/ 文件 | 合并策略 |
|-----------|-----------|---------|
| `op_kernel/dispatch_layout.h` | `op_kernel/dispatch_layout.h` | 保留 A3 版本，A2 分支通过 TilingKey 选择 |
| `op_kernel/dispatch_layout_a2.h` | `op_kernel/dispatch_layout_a2.h` | 以 ops2 版本为准 (含 tokenIdx 输出) |
| `op_kernel/notify_dispatch.h` | `op_kernel/notify_dispatch.h` | 保留一份，通过 `#ifdef` 区分 |
| `op_kernel/cam_moe_dispatch_normal.h` | `op_kernel/cam_moe_dispatch_normal.h` | 同上 |
| `op_kernel/cam_moe_combine_normal.h` | `op_kernel/cam_moe_combine_normal.h` | 同上 |
| `op_kernel/comm_args.h` | `op_kernel/comm_args.h` | 合并为一份 (ops2 版本含额外常量) |
| `op_kernel/sync_collectives.h` | `op_kernel/sync_collectives.h` | 合并为一份 |