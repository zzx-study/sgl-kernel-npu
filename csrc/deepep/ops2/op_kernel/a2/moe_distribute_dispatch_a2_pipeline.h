#ifndef MOE_DISTRIBUTE_DISPATCH_A2_PIPELINE_H
#define MOE_DISTRIBUTE_DISPATCH_A2_PIPELINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../cam_moe_distribute_dispatch_tiling.h"
#include "../moe_distribute_base.h"
#include "../comm_args.h"

namespace MoeDistributeDispatchA2Impl {
constexpr uint32_t STATE_OFFSET = 512;                 // 状态空间偏移地址
constexpr uint32_t STATUS_SIZE_LAYERED = 1024 * 1024;  // 1M
constexpr uint32_t HCCS_RING_BUFFER_HEAD_TAIL = 8 * 2 * 32;
constexpr uint32_t EACH_HCCS_RING_BUFFER_HEAD_TAIL = 2 * 32; // 两个元素
constexpr uint32_t RING_BUFFER_HEAD_TAIL = 2 * 32;
constexpr uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;  // 本卡状态空间偏移地址
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t INFO_NUM_IN_TOKENSTRUCK = 4;  // 在Token后加入4种信息:expIds, weights, tokenIdx, scales;
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint32_t PER_MSG_RDMA_SEND_TIME = 2;
constexpr uint32_t B32_PER_BLOCK = 8;
constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = 2 * UB_32B_ALIGN;
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 80 * 1024;
constexpr uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 64 * 32;
constexpr uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
constexpr uint32_t IPC_TOKEN_CNT_FLAG_OFFSET = 2 * 1024 * 1024 + 2 * UB_32B_ALIGN;
constexpr int32_t IPC_TOKEN_CNT_FLAG_WAIT = 0;
constexpr int32_t IPC_TOKEN_CNT_FLAG_READY = 1;
constexpr int32_t IPC_TOKEN_CNT_FLAG_FINISH = 2;
constexpr uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
constexpr uint32_t NOTIFY_OFFSET = 0 * 1024 * 1024;
constexpr uint32_t IPC_BUFF_ALIGN = 512;
constexpr uint32_t TOKEN_COUNT_SIZE = 32;
constexpr uint32_t MAX_SERVER_NUM = 32;
constexpr uint32_t RDMA_CHUNK = 32;
constexpr uint32_t FLAG_U32_CNT = TOKEN_COUNT_SIZE / 4;
constexpr int32_t IPC_FLAG_STEP_1 = 1ULL;
constexpr int32_t IPC_FLAG_STEP_2 = 2ULL;
constexpr uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
constexpr uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
constexpr uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;  // normal/low_latency dispatch&combine的预留大小一致
constexpr uint32_t EXTRA_TOKEN_INFO_NUM = 4U;              // 专家信息 权重信息 量化Scale 到达标志位
constexpr uint32_t BITS32_PER_BLOCK = 8U;
constexpr static uint32_t BW_ITEM_SIZE = 32;
constexpr int32_t FLAG_VALUE = -1;
constexpr uint32_t BS_UPPER = 4096;

constexpr uint32_t RDMA_HCCS_FORWARDER = 1;
constexpr uint32_t RDMA_SENDER = 2;
constexpr uint32_t FORWARDER_COORDINATOR = 3;
constexpr uint32_t HCCS_RECEIVER = 4;
constexpr uint32_t RDMA_COORDINATOR = 5;
constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 32;

constexpr uint32_t WAIT_HCCS_WRITE = 0;
constexpr uint32_t HCCS_SEND_END = 1;

#define TemplateMC2TypeA2PipelineClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2PipelineFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist
#define printflag(ss)                                                      \
    if (true) {                                       \
        printf("========rank:%d coreIdx:%d " #ss "\n", rankId_, aivId_); \
    }
using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeA2PipelineClass>
class MoeDistributeDispatchA2Pipeline
{
    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

public:
    __aicore__ inline MoeDistributeDispatchA2Pipeline(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales,
                                GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt, GM_ADDR epRankTokenCnt,
                                GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR tokenIdxPerExpert,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR expandScales, GM_ADDR syncCore,
                                GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM);
    __aicore__ inline void Process();
    template <AscendC::HardEvent event>
    __aicore__ inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

private:
    __aicore__ inline void CoreRoleAssign();
    __aicore__ inline void WorkBeforePipeline();
    __aicore__ inline void PrepareRdmaSend();
    __aicore__ inline void TriggerRdmaSend();
    __aicore__ inline void Rdma2HCCS();
    __aicore__ inline void CreditRecycle();
    __aicore__ inline void HCCS2Out();
    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value);
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr,
                            uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGt;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint8_t> readTokensU8Tensor_;
    GlobalTensor<uint8_t> rdmaSendU8Tensor_;
    GlobalTensor<uint8_t> rdmaRecvU8Tensor_;
    GlobalTensor<uint8_t> hccsRecvU8Tensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<uint32_t> expertToServerGlobalTensor_;
    GlobalTensor<int32_t> readStatusTensor_;
    GlobalTensor<int32_t> tokenServerIdxGMTensor_;
    GlobalTensor<int32_t> tokenServerCntGMTensor_;
    GlobalTensor<uint8_t> rdmaSendRingU8Tensor_;
    GlobalTensor<uint32_t> rdmaSendRingU32Tensor_;
    GlobalTensor<uint8_t> rdmaRecvRingU8Tensor_;
    GlobalTensor<uint32_t> rdmaRecvRingU32Tensor_;
    GlobalTensor<uint8_t> hccsRecvRingU8Tensor_;
    GlobalTensor<int32_t> rdmaRecvHeadTailTensor_;
    GlobalTensor<int32_t> rdmaSendHeadTailTensor_;
    GlobalTensor<int32_t> tokenCntTensor_;

    GlobalTensor<int32_t> epRankTokenCntGMTensor_;
    GlobalTensor<int32_t> srcOffsetRankTokenIdxGMTensor_;
    GlobalTensor<int32_t> dstOffsetRankTokenIdxGMTensor_;
    GlobalTensor<int32_t> tokenIdxPerExpertGMTensor_;
    GlobalTensor<int32_t> tokenPerRankGMTensor_;
    GlobalTensor<int32_t> syncCoreGMTensor_;
    GlobalTensor<uint32_t> hccsBufferStatusTensor_;

    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertToServerCntTensor_;
    LocalTensor<uint32_t> expertToServerIdxTensor_;

    LocalTensor<int32_t> tokenServerIdxTensor_;
    LocalTensor<int32_t> serverCountTensor_;
    LocalTensor<uint8_t> tokenStructInRdmaTensor_;
    LocalTensor<uint8_t> tokenStructInHccsTensor_;
    LocalTensor<uint8_t> rdmaUseTokenStructInHccsTensor_;
    LocalTensor<uint32_t> localHccsHeadTailTensor_;
    LocalTensor<int32_t> localSyncCoreTensor_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;

    TBuf<> tokenServerIdxBuf_;
    TBuf<> serverCountBuf_;

    TBuf<> expertCountBuf_;
    TBuf<> statusBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertToServerCntsBuf_;  // 总表，int类型只写1/0
    TBuf<> expertToServerIdxBuf_;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<> weightBuf_;
    TBuf<> tokenStructInRdmaBuf_;
    TBuf<> tokenStructInHccsBuf_;
    TBuf<> rdmaUseTokenStructInHccsBuf_;
    TBuf<> localHccsHeadTailBuf_;
    TBuf<> localSyncCoreBuf_;
    TBuf<> tempInt32Buf_;
    TBuf<> reduceBuf_;
    TBuf<> srcFloatBuf_;
    TBuf<> dstFloatBuf_;
    TBuf<> epRankTokenCntBuf_;
    TBuf<> rdmaInBuf_;
    TBuf<> rdmaInBuf2_;
    TBuf<> tmpBuf_;

    GM_ADDR expandXGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR weightsGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR dataBatchWriteInfo_;
    GM_ADDR expertToServerCntGM_;
    GM_ADDR shareAddrs[8];
    GM_ADDR shareAddrWins[8];
    GM_ADDR hccsHeadTailGM[8];
    __gm__ HcclAiRMAInfo *qp_info_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t kAlign_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0};  // aiv id
    uint32_t aivRole_{0};
    uint32_t moeExpertNum_{0};  // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumInServer_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t SERVER_SIZE_ON_WIN{0};
    uint32_t RANK_SIZE_ON_IPC{0};
    uint32_t WIN_SIZE{0};
    uint32_t bufferId_{0};
    uint32_t totalSize_{0};
    uint32_t totalWinSize_{0};
    uint32_t halfWinSize_{0};
    uint32_t serverNum{0};
    uint32_t rdmaItemNum{0};
    uint32_t hccsItemNum{0};
    uint32_t expertTokenNumsType_{0};
    int32_t rdmaSendHead[MAX_SERVER_NUM] = {0};
    int32_t rdmaSendTail[MAX_SERVER_NUM] = {0};
    uint32_t rdmaRecvHead[MAX_SERVER_NUM] = {0};
    uint32_t rdmaRecvTail[MAX_SERVER_NUM] = {0};
    uint32_t hccsRecvHead[MAX_SERVER_NUM] = {0};
    uint32_t hccsRecvTail[MAX_SERVER_NUM] = {0};
    int32_t dstServerCnt[MAX_SERVER_NUM] = {0};
    uint32_t senderNum{0};
    uint32_t triggerNum{0};
    // TokenStruck相关
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t srcRankInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    uint64_t magicVal_{0};
    // 当前server处理的专家范围
    uint32_t combineInnerCntOffset;
    uint32_t combineInnerCntIndexOffset;
    uint32_t combineOuterCntOffset;
    uint32_t combineOuterCntIndexOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo *)QpInfo)->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ *)(((__gm__ HcclAiRMAInfo *)QpInfo)->sqPtr +
                               destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo *)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo *)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t *)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t *)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;
    PipeBarrier<PIPE_ALL>();
    printf("RANK %d AIVID %d qpNum %d qp_ctx_entry %p sqBaseAddr %p \n", rankId_, aivId_, qpNum, qp_ctx_entry, sqBaseAddr);

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while (1) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();  // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                      // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);  // [7] owner_bit
    byte_4 |= 1U << 8U;                        // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t *)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)destDmaAddr;  // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;  // L_Key
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) =
        (uint64_t)srcDmaAddr;  // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;                     // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL;                           // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL;           // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL;  // [48:50] DB_SL = qp.sl

    __gm__ uint64_t *doorBellAddr = (__gm__ uint64_t *)(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();
    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();
    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt,
    GM_ADDR epRankTokenCnt, GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR tokenIdxPerExpert,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR expandScales, GM_ADDR syncCore, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(CamMoeDistributeDispatchA2TilingData);
    auto tiling = (__gm__ CamMoeDistributeDispatchA2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);
    qp_info_ = (__gm__ HcclAiRMAInfo *)(((__gm__ HcclA2CombineOpParam *)contextGM0)->aiRMAInfo);

    winContext_ = (__gm__ HcclOpResParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    aivId_ = GetBlockIdx();
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);
    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    globalBs_ = tilingData.moeDistributeDispatchInfo.globalBs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;
    kAlign_ = RoundUp(axisK_, (uint32_t)8);
    totalSize_ = winContext_->winSize;
    totalWinSize_ = RDMA_DATA_SIZE;  // RDMA 800 MB空间, 与low_latency一致
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - serverNum * RING_BUFFER_HEAD_TAIL;
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;  // 共享内存上每个server块的大小

    // struce相关信息初始化计算
    tokenStructLen_ = axisH_ * sizeof(ExpandXOutType) +
                        INFO_NUM_IN_TOKENSTRUCK * (kAlign_ * sizeof(uint32_t)); // token和四元组大小
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);                                           // 纯token大小
    expLenInStruct_ = kAlign_ * sizeof(int32_t);                                                  // topkId大小
    weightLenInStruct_ = kAlign_ * sizeof(uint32_t);                                               // weight大小
    cntLenInStruct_ = kAlign_ * sizeof(uint32_t);                                                  // tokenIdx大小
    realLenInStruct_ = axisK_ * sizeof(uint32_t);                 // 内存中实际有效部分，跟 axisK_ 有关
    expOffsetInStruct_ = tokenLenInStruct_;                       // 开始写topkId的起始位置
    weightOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_;  // 开始写weight的起始位置
    cntOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_;  // 开始写tokenIdx的起始位置
    scaleOffsetInStruct_ =
        tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + cntLenInStruct_;  // 开始写scales的起始位置
    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    rdmaItemNum = (SERVER_SIZE_ON_WIN - EXP_TOKEN_COUNT_FLAG_CNT) / tokenStructLen_;
    printf("RANK%d AIVID%d rdmaItemNum:%d SERVER_SIZE_ON_WIN:%d tokenStructLen_:%d\n",
    rankId_, aivId_, rdmaItemNum, SERVER_SIZE_ON_WIN, tokenStructLen_);

    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (SERVER_RANK_SIZE);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;
    hccsItemNum = RANK_SIZE_ON_IPC / tokenStructLen_;

    expertIdsCnt_ = axisBS_ * axisK_;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + totalWinSize_ +
            NOTIFY_OFFSET));
        shareAddrWins[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + NOTIFY_OFFSET +
            halfWinSize_ * bufferId_));
    }
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;

    tokenServerIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerIdx);
    tokenServerCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerCnt);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    epRankTokenCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)epRankTokenCnt);
    srcOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)srcOffsetRankTokenIdx);
    dstOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstOffsetRankTokenIdx);
    tokenIdxPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenIdxPerExpert);

    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    weightsOutGt.SetGlobalBuffer((__gm__ float *)(expandScales));
    syncCoreGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(syncCore));

    rdmaSendRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    rdmaRecvRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_));
    rdmaSendRingU32Tensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_));
    rdmaRecvRingU32Tensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_));
    //hccsRecvRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_ + halfWinSize_ / 2));
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        hccsHeadTailGM[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + totalWinSize_));
    }
    hccsBufferStatusTensor_.SetGlobalBuffer((__gm__ uint32_t *)
        reinterpret_cast<uint64_t>(hccl_.GetWindowsInAddr(rankId_) + totalWinSize_ + HCCS_RING_BUFFER_HEAD_TAIL));
    hccsBufferStatusTensor_.SetValue(0, WAIT_HCCS_WRITE);
    DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>
            (hccsBufferStatusTensor_);
    rdmaRecvHeadTailTensor_.SetGlobalBuffer((__gm__ int32_t *)
                        (windowInGM_ + WIN_SIZE));
    rdmaSendHeadTailTensor_.SetGlobalBuffer((__gm__ int32_t *)
                        (windowOutGM_ + WIN_SIZE));
    expertTokenNumsOutGM_ = expertTokenNumsOut;  // 无GlobalTensor
    epRecvCountsGM_ = epRecvCountsOut;           // 无GlobalTensor
    statusSpaceGm_ = windowInGM_ + WIN_SIZE;

    expandXGM_ = x;
    expandIdxGM_ = expertIds;
    weightsGM_ = expertScales;

    dataBatchWriteInfo_ = workspaceGM;
    dataBatchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint64_t *)(dataBatchWriteInfo_),
                                              serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK);
    expertToServerCntGM_ = dataBatchWriteInfo_ + serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK * sizeof(uint64_t);
    expertToServerGlobalTensor_.SetGlobalBuffer((__gm__ uint32_t *)(expertToServerCntGM_),
                                                RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));

    tokenCntTensor_.SetGlobalBuffer((__gm__ int32_t *)reinterpret_cast<uint64_t>
            (hccl_.GetWindowsInAddr(rankId_) + totalWinSize_ + IPC_TOKEN_CNT_OFFSET));

    combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int32_t);
    combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t);
    combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;
    tpipe_->InitBuffer(batchWriteInfoBuf_, PER_MSG_RDMA_SEND_TIME * BW_ITEM_SIZE);  // 2 * 32

    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
    batchWriteU32Tensor_ = batchWriteU64Tensor_.template ReinterpretCast<uint32_t>();

    tpipe_->InitBuffer(statusBuf_, 2 * UB_32B_ALIGN);  // 32

    tpipe_->InitBuffer(expertToServerIdxBuf_, serverNum * sizeof(uint32_t));  // rankSize / 8 * 4
    expertToServerIdxTensor_ = expertToServerIdxBuf_.Get<uint32_t>();

    tpipe_->InitBuffer(tokenStructInRdmaBuf_, tokenStructLen_);
    tokenStructInRdmaTensor_ = tokenStructInRdmaBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(tokenStructInHccsBuf_, tokenStructLen_);
    tokenStructInHccsTensor_ = tokenStructInHccsBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(rdmaUseTokenStructInHccsBuf_, tokenStructLen_);
    rdmaUseTokenStructInHccsTensor_ = rdmaUseTokenStructInHccsBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(localHccsHeadTailBuf_, EACH_HCCS_RING_BUFFER_HEAD_TAIL);
    localHccsHeadTailTensor_ = localHccsHeadTailBuf_.Get<uint32_t>();

    tpipe_->InitBuffer(expertCountBuf_, moeExpertNum_ * sizeof(int32_t));  // moeNum * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertCountTensor_, 0, moeExpertNum_);
    tpipe_->InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);  // 176K
    tpipe_->InitBuffer(weightBuf_, UB_32B_ALIGN);      // 32
    tpipe_->InitBuffer(localSyncCoreBuf_, aivNum_ * DEFAULT_SYNCALL_NEED_SIZE); 
    localSyncCoreTensor_ = localSyncCoreBuf_.Get<int32_t>();
    Duplicate<int32_t>(localSyncCoreTensor_, 0, aivNum_);
    tpipe_->InitBuffer(tempInt32Buf_, UB_32B_ALIGN);
    tpipe_->InitBuffer(srcFloatBuf_, moeExpertNum_ * worldSize_ * sizeof(float));
    tpipe_->InitBuffer(dstFloatBuf_, worldSize_ * sizeof(float));
    tpipe_->InitBuffer(reduceBuf_, moeExpertNum_ * worldSize_ * sizeof(float));
    tpipe_->InitBuffer(epRankTokenCntBuf_, moeExpertNum_ * worldSize_ * sizeof(int32_t));
    tpipe_->InitBuffer(rdmaInBuf_, UB_32B_ALIGN);
    tpipe_->InitBuffer(rdmaInBuf2_, UB_32B_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();
    tpipe_->InitBuffer(serverCountBuf_, serverNum * sizeof(int32_t));
    serverCountTensor_ = serverCountBuf_.Get<int32_t>();
    tpipe_->InitBuffer(tmpBuf_, 3 * UB_32B_ALIGN);

    CoreRoleAssign();
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    int32_t state = selfStatusTensor(aivId_ * UB_32B_ALIGN);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        for (int i = 0; i < serverNum; i++) {
            rdmaRecvHeadTailTensor_.SetValue(RING_BUFFER_HEAD_TAIL * i / sizeof(int32_t), FLAG_VALUE);
            DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                rdmaRecvHeadTailTensor_[RING_BUFFER_HEAD_TAIL * i]);
            // printf("RANK %d AIVID %d 540 Init serverId %d rdmaTail %d rdmaRecvHeadTailTensor_ %d\n", rankId_, aivId_, i,
            // *(__gm__ int32_t *)(hccl_.GetWindowsInAddr(rankId_) +
            // halfWinSize_ * bufferId_ + WIN_SIZE + RING_BUFFER_HEAD_TAIL * i),
            // rdmaRecvHeadTailTensor_(RING_BUFFER_HEAD_TAIL * i / sizeof(int32_t)));
        }
    AscendC::DumpTensor(rdmaRecvHeadTailTensor_, 546, 128);
    }

    LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
    // 每次调用magic++,用来区分不同轮次
    GlobalTensor<uint64_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ uint64_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
                            aivId_ * UB_32B_ALIGN / sizeof(uint64_t));
    DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    tempLocal(0) += 1ULL;
    magicVal_ = tempLocal(0);
    DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    if (aivId_ == 0) {
        for (int i = 0; i < SERVER_RANK_SIZE * 2; i++) {
            printf("Dispatch Init [RANK %d AIC %d] windowInGM_[%d]: %p windowOutGM_[%d]:%p RANK_SIZE_ON_IPC:%d totalSize_:%d windowInGM_end[%d]:%p\n",
                rankId_, aivId_, i, hccl_.GetWindowsInAddr(i), i, hccl_.GetWindowsOutAddr(i), RANK_SIZE_ON_IPC, totalSize_, i, hccl_.GetWindowsInAddr(i) + totalSize_);
        }
    }
}

// 完成流水线之前的准备工作，提前写入发往每个server的token数
template <TemplateMC2TypeA2PipelineClass>
__aicore__ void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::WorkBeforePipeline()
{
    uint32_t needProcessNum = serverNum / aivNum_;
    uint32_t restServerNum = serverNum % aivNum_;
    uint32_t startServerId = aivId_ * needProcessNum;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    if (aivId_ < restServerNum) {
        needProcessNum += 1;
        startServerId += aivId_;
    } else {
        startServerId += restServerNum;
    }
    DataCopyExtParams serverCountParams = {1U, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams tmpParams = {1U, UB_32B_ALIGN, 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> tmpPadExtParams{false, 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> tmpUintPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams tmpCntFlagParams = {1U, 3 * UB_32B_ALIGN, 0U, 0U, 0U};
    DataCopyPad(serverCountTensor_, tokenServerCntGMTensor_, serverCountParams, copyPadExtParams);
    LocalTensor<int32_t> tmpTensor = tmpBuf_.Get<int32_t>();
    if (aivId_ == 0) {
        tokenCntTensor_.SetValue(0 * UB_32B_ALIGN / sizeof(uint32_t), static_cast<int32_t>(axisBS_));
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(tokenCntTensor_[0 * UB_32B_ALIGN / sizeof(uint32_t)]);
        tokenCntTensor_.SetValue(2 * UB_32B_ALIGN / sizeof(uint32_t), IPC_TOKEN_CNT_FLAG_READY);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(tokenCntTensor_[2 * UB_32B_ALIGN / sizeof(uint32_t)]);
    }
    SyncAll<true>();
    if (aivId_ < localRankId) {
        GlobalTensor<int32_t> preRankTokenTmpCntTensor;
        preRankTokenTmpCntTensor.SetGlobalBuffer((__gm__ int32_t *)reinterpret_cast<uint64_t>
            (hccl_.GetWindowsInAddr(aivId_ + (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE) +
            totalWinSize_ + IPC_TOKEN_CNT_OFFSET));
        DataCopyPad(tmpTensor, preRankTokenTmpCntTensor, tmpCntFlagParams, tmpPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        while (tmpTensor(2 * UB_32B_ALIGN / sizeof(uint32_t)) < IPC_TOKEN_CNT_FLAG_READY) {
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            DataCopyPad(tmpTensor, preRankTokenTmpCntTensor, tmpCntFlagParams, tmpPadExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            continue;
        }
        printf("RANK%d AIVID%d localRankToken:%d \n",
            rankId_, aivId_, tmpTensor(0));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::SetAtomicAdd<int32_t>();
        DataCopyPad(tokenCntTensor_[1 * UB_32B_ALIGN / sizeof(int32_t)],
            tmpTensor[0 * UB_32B_ALIGN / sizeof(int32_t)], tmpParams);
        AscendC::SetAtomicNone();
        PipeBarrier<PIPE_MTE3>();
    }
    SyncAll<true>();
    if (aivId_ == 0) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        tokenCntTensor_.SetValue(2 * UB_32B_ALIGN / sizeof(uint32_t), IPC_TOKEN_CNT_FLAG_FINISH);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(tokenCntTensor_[ 2 * UB_32B_ALIGN / sizeof(int32_t)]);
    }
    printf("RANK%d AIVID%d tokenCntTensor_%d Syncflag:%d\n",
        rankId_, aivId_,
        tokenCntTensor_(1 * UB_32B_ALIGN / sizeof(int32_t)),
        tokenCntTensor_(2 * UB_32B_ALIGN / sizeof(int32_t)));
    LocalTensor<uint32_t> tmp1Tensor = tmpBuf_.Get<uint32_t>();
    for (uint32_t dstServerId = startServerId; dstServerId < startServerId + needProcessNum; ++dstServerId) {
        LocalTensor<uint32_t> writeCntLt = tBuf.GetWithOffset<uint32_t>
                                            (FLAG_U32_CNT, 0);
        writeCntLt.SetValue(0, serverCountTensor_(dstServerId));
        uint32_t destOffset = (dstServerId * SERVER_SIZE_ON_WIN) / sizeof(uint32_t);
        GlobalTensor<uint32_t> sumTokenCntServerTensor;
        sumTokenCntServerTensor.SetGlobalBuffer((__gm__ uint32_t *)reinterpret_cast<uint64_t>
            (hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + SERVER_RANK_SIZE - 1) +
            totalWinSize_ + IPC_TOKEN_CNT_OFFSET)); // 取server内的最后一张卡，用来计算本server内的 token 总和
        // 最后一张卡的本卡的token数与前序卡的token数相加
        DataCopyPad(tmp1Tensor, sumTokenCntServerTensor, tmpCntFlagParams, tmpUintPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        while (tmp1Tensor.GetValue(2 * UB_32B_ALIGN / sizeof(uint32_t)) < IPC_TOKEN_CNT_FLAG_FINISH) {
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            DataCopyPad(tmp1Tensor, sumTokenCntServerTensor, tmpCntFlagParams, tmpUintPadExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            continue;
        }
        tmp1Tensor(2 * UB_32B_ALIGN / sizeof(uint32_t)) = tmp1Tensor(0 * UB_32B_ALIGN / sizeof(uint32_t))
            + tmp1Tensor(1 * UB_32B_ALIGN / sizeof(uint32_t));
        printf("WorkBeforePipeline RANK%d AIVID%d single:%d preSum:%d flag:%d\n",
                rankId_, aivId_,
                tmp1Tensor(0), tmp1Tensor(1 * UB_32B_ALIGN / sizeof(uint32_t)),
                tmp1Tensor(2 * UB_32B_ALIGN / sizeof(uint32_t)));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        if (dstServerId == rankId_ / SERVER_RANK_SIZE) {
            DataCopyPad(rdmaRecvRingU32Tensor_[destOffset], writeCntLt, tmpParams);
            DataCopyPad(rdmaRecvRingU32Tensor_[destOffset + UB_32B_ALIGN / sizeof(uint32_t)],
                                tmp1Tensor[2 * UB_32B_ALIGN / sizeof(uint32_t)], tmpParams);
        } else {
            DataCopyPad(rdmaSendRingU32Tensor_[destOffset], writeCntLt, tmpParams);
            DataCopyPad(rdmaSendRingU32Tensor_[destOffset + UB_32B_ALIGN / sizeof(uint32_t)],
                                tmp1Tensor[2 * UB_32B_ALIGN / sizeof(uint32_t)], tmpParams);
        }
    }
    SyncAll<true>();
}

// 分配各个核的角色，各个角色负责不同的任务
template <TemplateMC2TypeA2PipelineClass>
__aicore__ void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::CoreRoleAssign()
{
    senderNum = aivNum_ / 4 - 1;
    aivRole_ = aivId_ / senderNum + 1;
    triggerNum = aivNum_ - 4 * senderNum;
}

// 由RDMA_SENDER执行，负责将要发送的token数据进行筛选打包，装载到rdmaSendRingU8Tensor这个环形buffer上，并更新headTailTensor中环形buffer头和尾的数据
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::PrepareRdmaSend()
{
    if (aivRole_ != RDMA_SENDER) {
        return;
    }
    //printflag(438);
    uint32_t localIndex = aivId_ % senderNum;
    if (localIndex >= serverNum) {
        return;
    }
    //printflag(443);
    uint32_t sendTokenNum = axisBS_;
    uint32_t startTokenId = 0;
    uint32_t endTokenId = axisBS_;
    int32_t expertId = 0;
    uint32_t dstServerId = 0;
    uint32_t tokenIndex = 0;
    uint32_t startMutexIndex = 0;
    uint32_t tokenUbSize = tokenStructLen_;
    if constexpr (DynamicQuant || StaticQuant) {
        tokenUbSize = axisH_ * sizeof(XType) + INFO_NUM_IN_TOKENSTRUCK * (kAlign_ * sizeof(uint32_t));
    }
    tpipe_->InitBuffer(tokenServerIdxBuf_, sendTokenNum * serverNum * sizeof(int32_t));

    tokenServerIdxTensor_ = tokenServerIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> tmpInt32Tensor = tempInt32Buf_.Get<int32_t>();
    LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
    DataCopyExtParams tokenServerIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * serverNum * sizeof(int32_t)), 0U,
                                              0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(tokenServerIdxTensor_, tokenServerIdxGMTensor_[startTokenId * serverNum], tokenServerIdxParams,
                copyPadExtParams);
    // 这几个tensor是相同的地址空间，只是数据类型不一样，用于组装要发送的一个token和后面的metadata
    LocalTensor<uint8_t> tokenTempTensorU8_ =
        tBuf.GetWithOffset<uint8_t>(((tokenUbSize) / sizeof(uint8_t)), TBUF_TEMP_OFFSET);
    LocalTensor<uint32_t> tokenTempTensorU32_ =
        tBuf.GetWithOffset<uint32_t>(((tokenUbSize) / sizeof(uint32_t)), TBUF_TEMP_OFFSET);
    LocalTensor<XType> tokenLt = tBuf.GetWithOffset<XType>(((tokenUbSize) / sizeof(XType)), TBUF_TEMP_OFFSET);
    GlobalTensor<uint8_t> xGMTensorU8_;
    xGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandXGM_);
    GlobalTensor<uint8_t> expertIdsGMTensorU8_;
    expertIdsGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandIdxGM_);

    GlobalTensor<uint32_t> expertIdsGMTensorU32_;
    expertIdsGMTensorU32_.SetGlobalBuffer((__gm__ uint32_t *)expandIdxGM_);

    GlobalTensor<uint8_t> weightGt;
    weightGt.SetGlobalBuffer((__gm__ uint8_t *)weightsGM_);

    DataCopyExtParams tokenCopyParamsQuant{1, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0, 0, 0};
    DataCopyExtParams tokenCopyParamsNoQuant{static_cast<uint16_t>(1), static_cast<uint16_t>(tokenLenInStruct_), 0, 0,
                                             0};
    DataCopyPadExtParams<uint8_t> tokenPadParams;

    DataCopyExtParams expCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(realLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> expPadParams;

    DataCopyExtParams weightCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(realLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> weightPadParams;

    DataCopyExtParams tmpInt32CopyParams{static_cast<uint16_t>(1), UB_32B_ALIGN, 0, 0, 0};
    DataCopyPadExtParams<int32_t> tmpInt32PadParams{false, 0, 0, 0};

    for (int i = 0; i < sendTokenNum; i++) {
        if constexpr (DynamicQuant || StaticQuant) {
            DataCopyPad(tokenTempTensorU8_, xGMTensorU8_[(startTokenId + i) * axisH_ * sizeof(XType)],
                        tokenCopyParamsQuant, tokenPadParams);
            LocalTensor<float> tokenCastLt = tBuf.GetWithOffset<float>(
                ((axisH_ * sizeof(float)) / sizeof(float)), RoundUp(TBUF_TEMP_OFFSET + tokenUbSize, B32_PER_BLOCK));
            QuantProcess(1, tokenLt, tokenCastLt);
        } else {
            DataCopyPad(tokenTempTensorU8_, xGMTensorU8_[(startTokenId + i) * tokenLenInStruct_],
                        tokenCopyParamsNoQuant, tokenPadParams);
        }
        // 拷贝topkIds 可省略
        DataCopyPad(tokenTempTensorU8_[expOffsetInStruct_], expertIdsGMTensorU8_[(startTokenId + i) * realLenInStruct_],
                    expCopyParams, expPadParams);

        // 拷贝weight
        DataCopyPad(tokenTempTensorU8_[weightOffsetInStruct_], weightGt[(startTokenId + i) * realLenInStruct_],
                    weightCopyParams, weightPadParams);

        tokenTempTensorU32_.SetValue(cntOffsetInStruct_ / sizeof(uint32_t),
                    startTokenId + i + static_cast<uint32_t>(tokenCntTensor_(1 * UB_32B_ALIGN / sizeof(uint32_t))));
        printf("PrepareRdmaSend RANK%d AIVID%d tokenCntTensor_%d\n",
                    rankId_, aivId_, tokenCntTensor_(1 * UB_32B_ALIGN / sizeof(uint32_t)));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (int j = localIndex; j < localIndex + 1; j++) {
            if (tokenServerIdxTensor_(i * serverNum + j) == -1) {
                continue;
            }
            if (j == rankId_ / SERVER_RANK_SIZE) {
                DataCopyPad(tmpInt32Tensor, rdmaRecvHeadTailTensor_[j * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)],
                            tmpInt32CopyParams, tmpInt32PadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                int32_t rdmaRecvTail = tmpInt32Tensor.GetValue(0);
                rdmaRecvTail = rdmaRecvTail == -1 ? 0 : rdmaRecvTail;
                uint32_t destOffset =
                    j * SERVER_SIZE_ON_WIN + tokenStructLen_ * (rdmaRecvTail % rdmaItemNum) + EXP_TOKEN_COUNT_FLAG_CNT;
                DataCopy(rdmaRecvRingU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenUbSize / sizeof(uint8_t));
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                tmpInt32Tensor(0) = rdmaRecvTail + 1;
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                DataCopyPad(rdmaRecvHeadTailTensor_[j * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)], tmpInt32Tensor,
                            tmpInt32CopyParams);
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            } else {
                DataCopyPad(tmpInt32Tensor, rdmaSendHeadTailTensor_[j * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)],
                            tmpInt32CopyParams, tmpInt32PadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                int32_t rdmaSendTail = tmpInt32Tensor.GetValue(0);
                uint32_t destOffset =
                    j * SERVER_SIZE_ON_WIN + tokenStructLen_ * (rdmaSendTail % rdmaItemNum) + EXP_TOKEN_COUNT_FLAG_CNT;
                DataCopy(rdmaSendRingU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenUbSize / sizeof(uint8_t));
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                tmpInt32Tensor(0) = rdmaSendTail + 1;
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(rdmaSendHeadTailTensor_[j * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)], tmpInt32Tensor,
                            tmpInt32CopyParams);
                printf("RANK %d AIVID %d 802 rdmaHead %d rdmaSendHeadTailTensor_ %d\n", rankId_, aivId_, tmpInt32Tensor.GetValue(0), rdmaSendHeadTailTensor_(j * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)));
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    // for token in assigned_tokens:
    //     if token_dst_rank == my_rank:
    //         wait_until(rdma_tail - rdma_head < capacity)
    //         pack_token(token, rdma_buffer[rdma_tail])
    //         rdma_tail++
}

// 由RDMA_COORDINATOR执行，负责触发准备好的RDMA发送任务，发送到rdmaRecvRingU8Tensor这个环形buffer上，并更新环形buffer中头和尾的值
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::TriggerRdmaSend()
{
    if (aivRole_ != RDMA_COORDINATOR) {
        return;
    }
    uint32_t localIndex = aivId_ % triggerNum;
    uint32_t destServerNum = serverNum / triggerNum;  // 每个AIV要处理的server数
    int32_t finishedServer = 0;
    uint32_t rdmaServerNum = serverNum % triggerNum;
    uint32_t startServerId = destServerNum * localIndex;
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;  // 当前serverId
    if (localIndex < rdmaServerNum) {                   // 前remainderRankNum个aiv需要多发1个卡的数据
        destServerNum += 1;
        startServerId += localIndex;
    } else {
        startServerId += rdmaServerNum;
    }
    if (destServerNum == 0) {
        return;
    }
    uint32_t endServerId = startServerId + destServerNum;
    bool isSendFlag = true;

    int32_t sendTail = -1;
    int32_t sendHead = -1;
    int32_t oldSendHead = -1;
    int32_t oldSendTail = -1;
    LocalTensor<int32_t> status = statusBuf_.Get<int32_t>();
    DataCopyExtParams statusParams{1, UB_32B_ALIGN, 0, 0, 0};
    DataCopyPadExtParams<int32_t> statusPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams serverCountParams = {1U, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    printf("RANK %d AIVID %d 845 TriggerRdmaSend destServerNum %d\n", rankId_, aivId_, destServerNum);
    while (finishedServer < destServerNum) {
        // 当前aiv负责 [startServerId,endServerId) 个 server
        for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
            DataCopyPad(serverCountTensor_, tokenServerCntGMTensor_, serverCountParams, copyPadExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            // 往本机发送的token在PrepareRdmaSend已经写入rdmaRacv的ring buffer，此处不触发rdma发送，设置为 0
            dstServerCnt[dstServerInd] = dstServerInd == curServerId ? 0 : serverCountTensor_(dstServerInd);
            if (dstServerCnt[dstServerInd] <= 0) {
                continue;
            }
            printf("RANK %d AIVID %d 852 TriggerRdmaSend dstServerCnt %d dstServerInd %d \n",
                rankId_, aivId_, dstServerCnt[dstServerInd], dstServerInd);
            uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + dstServerInd * SERVER_RANK_SIZE;  // 目标Rank
            rdmaSendHead[dstServerInd] =
                rdmaSendHeadTailTensor_.GetValue(dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8);
            rdmaSendTail[dstServerInd] =
                rdmaSendHeadTailTensor_.GetValue(dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t));
            PipeBarrier<PIPE_ALL>();
            uint32_t sendNum = RDMA_CHUNK < dstServerCnt[dstServerInd] ? RDMA_CHUNK : dstServerCnt[dstServerInd];
            if ASCEND_IS_AIV {
                printf("RANK %d AIVID %d wait while\n", rankId_, aivId_);
                while (rdmaSendTail[dstServerInd] - rdmaSendHead[dstServerInd] < sendNum) {
                    rdmaSendHead[dstServerInd] =
                        rdmaSendHeadTailTensor_.GetValue(dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8);
                    rdmaSendTail[dstServerInd] = 
                        rdmaSendHeadTailTensor_.GetValue(dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t));
                    continue;
                }
                sendHead = rdmaSendHead[dstServerInd] % rdmaItemNum;
                sendTail = rdmaSendTail[dstServerInd] % rdmaItemNum;
                if (oldSendTail != sendTail || oldSendHead != sendHead) {
                    printf("TriggerRdmaSend [RANK%d AIVID %d] sendTail(%d) sendHead(%d)\n",
                    rankId_, aivId_, sendTail, sendHead);
                    oldSendTail = sendTail;
                    oldSendHead = sendHead;
                }
                __gm__ uint8_t * dstDataRdmaAddr = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(dstRankId) +
                            EXP_TOKEN_COUNT_FLAG_CNT + halfWinSize_ * bufferId_ + curServerId * SERVER_SIZE_ON_WIN +
                            sendTail * tokenStructLen_);
                // src卡GetWindowsInAddr地址, 要发给serverIndex，即是本端的rdma地址
                __gm__ uint8_t * srcDataRdmaAddr = (__gm__ uint8_t *)(hccl_.GetWindowsOutAddr(rankId_) +
                            EXP_TOKEN_COUNT_FLAG_CNT +  halfWinSize_ * bufferId_ + dstServerInd * SERVER_SIZE_ON_WIN +
                            sendTail * tokenStructLen_);
                // 去往该Server的传输的数据量
                uint32_t validDataLength = sendNum * tokenStructLen_;

                // 第一次发送需要发送本rank发往对端的token数以及本server内token数
                if (isSendFlag) {
                    dstDataRdmaAddr = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ +
                        curServerId * SERVER_SIZE_ON_WIN);
                    srcDataRdmaAddr = (__gm__ uint8_t *)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                        dstServerInd * SERVER_SIZE_ON_WIN);
                    validDataLength = EXP_TOKEN_COUNT_FLAG_CNT + sendNum * tokenStructLen_;
                    isSendFlag = false;
                }

                printf("RANK %d AIVID %d start to Rdma srcDataRdmaAddr %p dstDataRdmaAddr %p \n", rankId_, aivId_, srcDataRdmaAddr, dstDataRdmaAddr);
                AIVRDMAPostSend(srcDataRdmaAddr, dstDataRdmaAddr, dstRankId, validDataLength, qp_info_);
                printf("RANK %d AIVID %d 913 TriggerRdmaSend srcDataRdmaAddr %p dstDataRdmaAddr %p dstRankId %d validDataLength %d qp_info_ %p\n",
                        rankId_, aivId_, srcDataRdmaAddr, dstDataRdmaAddr, dstRankId, validDataLength, qp_info_);
                bufferChosenGlobal_(0) = bufferId_ ^ 1;
                DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(bufferChosenGlobal_);
                printf("RANK %d AIVID %d finish Rdma\n", rankId_, aivId_);
            }
        }
        printf("RANK %d AIVID %d 902 TriggerRdmaSend\n", rankId_, aivId_);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
            if (dstServerInd != rankId_ / SERVER_RANK_SIZE) {
                DataCopyPad(status, rdmaSendHeadTailTensor_[dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8], statusParams, statusPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                rdmaSendHead[dstServerInd] = status(0);
                printf("RANK %d AIVID %d 919 TriggerRdmaSend status %d finishedServer %d\n", rankId_, aivId_, status(0), finishedServer);
                rdmaSendHead[dstServerInd] += RDMA_CHUNK < dstServerCnt[dstServerInd] ?
                                            RDMA_CHUNK : dstServerCnt[dstServerInd];
                dstServerCnt[dstServerInd] -= RDMA_CHUNK < dstServerCnt[dstServerInd] ?
                                            RDMA_CHUNK : dstServerCnt[dstServerInd];
                status(0) = rdmaSendHead[dstServerInd];
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                DataCopyPad(rdmaSendHeadTailTensor_[dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8], status, statusParams);
                uint32_t destRankId = rankId_ % SERVER_RANK_SIZE + dstServerInd * SERVER_RANK_SIZE;
                __gm__ uint8_t * srcFlagAddr = (__gm__ uint8_t *)(hccl_.GetWindowsOutAddr(rankId_) +
                        halfWinSize_ * bufferId_ + WIN_SIZE + dstServerInd * RING_BUFFER_HEAD_TAIL + UB_32B_ALIGN);
                __gm__ uint8_t * dstFlagAddr = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(destRankId) +
                        halfWinSize_ * bufferId_ + WIN_SIZE + curServerId * RING_BUFFER_HEAD_TAIL);
                printf("RANK %d AIVID %d AIVRDMAPostSend rdmaSendHead %d rdmaSendHeadTailTensor_ %d qp_info_ %p\n",
                rankId_, aivId_, rdmaSendHead[dstServerInd],
                rdmaSendHeadTailTensor_(dstServerInd * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8), qp_info_);
                AIVRDMAPostSend(srcFlagAddr, dstFlagAddr, destRankId, UB_32B_ALIGN, qp_info_);
                printf("RANK %d AIVID %d 942 srcFlagAddr %p dstFlagAddr %p\n", rankId_, aivId_, srcFlagAddr, dstFlagAddr);
            }
            finishedServer = dstServerCnt[dstServerInd] <= 0 ? finishedServer + 1 : finishedServer;
        }
    }printf("RANK %d AIVID %d after while\n", rankId_, aivId_);
    // AscendC::DumpTensor(readStatusTensor_, localIndex * 1000 + 694, 256);
    // printflag(TriggerRdmaSend695);
    // if local_tail - last_sent >= RDMA_CHUNK:
    //     roce_write(rdma_buffer[last_sent : local_tail])
    //     rdma_atomic_add(remote_tail, local_tail - last_sent)
    //     last_sent = local_tail
}

// 由RDMA_HCCS_FORWARDER执行，负责将RDMA通信中接收到的数据进行HCCS的再发送，发送到hccsRecvRingU8Tensor这个环形buffer上
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Rdma2HCCS()
{
    if (aivRole_ != RDMA_HCCS_FORWARDER) {
        return;
    }

    uint32_t serverId = rankId_ / SERVER_RANK_SIZE;
    uint32_t expertIdxStart = serverId * moeExpertNumInServer_;
    uint32_t expertIdxEnd = expertIdxStart + moeExpertNumInServer_;
    uint32_t senderNum = aivNum_ / 4 - 1;
    // 每个核负责一个hccs接收环形buffer
    uint32_t localWorkCoreId = aivId_ % senderNum;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    if (localWorkCoreId >= SERVER_RANK_SIZE) {
        return;
    }
    uint32_t preTokenCnt = 0;
    DataCopyExtParams tokenStructParams{1, static_cast<uint32_t>(tokenStructLen_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> tokenStructPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams statusParams{1, UB_32B_ALIGN, 0, 0, 0};
    DataCopyPadExtParams<int32_t> statusPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams tokenIdxParams{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> tokenIdxPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams epRankTokenCntParams{1, static_cast<uint32_t>(moeExpertNum_ * worldSize_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> epRankTokenCntPadParams{false, 0U, 0U, 0U};
    DataCopyParams hccsHesdTailParams{1, EACH_HCCS_RING_BUFFER_HEAD_TAIL, 0, 0};
    DataCopyParams hccsHesdTailSingleParams{1, UB_32B_ALIGN, 0, 0};
    uint32_t processedTokenNum = 0;
    DataCopyExtParams serverCountParams = {1U, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(serverCountTensor_, tokenServerCntGMTensor_[0], serverCountParams, copyPadExtParams);
    bool waitFalg = true;
    LocalTensor<int32_t> status = statusBuf_.Get<int32_t>();
    status(0) = -1;
    printf("RANK %d AIVID %d 965 Rdma2HCCS status %d\n", rankId_, aivId_, status(0));
    uint32_t count = 0;
    while (waitFalg) {
        waitFalg = false;
        for (int i = 0;i < serverNum; i++) {
            DataCopyPad(status, rdmaRecvHeadTailTensor_[i * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)], statusParams, statusPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            printf("RANK %d AIVID %d 967 Rdma2HCCS rdmaTail %d serverId %d rdmaRecvHeadTailTensor_ %d\n",
                rankId_, aivId_, status(0), i, rdmaRecvHeadTailTensor_.GetValue(i * RING_BUFFER_HEAD_TAIL / sizeof(int32_t)));
            if (status(0) == -1) {
                waitFalg = true;
                break;
            }
        }
        count++;
    }
    int32_t rdmaTail = -1;
    int32_t rdmaHead = -1;
    printf("Rdma2HCCS RANK%d AIVIDX%d 749 after while\n", rankId_, aivId_);
    // 当前aicore负责的hccs环形buffer接收的来自于各server的token数，即每个aicore需要处理的token
    uint32_t tokenGlobalCnt = 0;
    for (int i = 0; i < serverNum; i++) {
        tokenGlobalCnt += rdmaRecvRingU32Tensor_.GetValue(i * SERVER_SIZE_ON_WIN / sizeof(uint32_t));
    }
    printf("Rdma2HCCS RANK%d AIVID %d 780 tokenGlobalCnt:%d rdmaItemNum:%d processedTokenNum:%d\n",
    rankId_, aivId_, tokenGlobalCnt, rdmaItemNum, processedTokenNum);
    // 此处processedTokenNum不用原子加是因为就算当前aicore不处理本token也会遍历token并计数
    while (processedTokenNum < tokenGlobalCnt) {
        for (int i = 0; i < serverNum * rdmaItemNum; i++) {
            uint32_t currentServerId = i / rdmaItemNum;
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            rdmaTail = rdmaRecvHeadTailTensor_.GetValue(currentServerId * RING_BUFFER_HEAD_TAIL / sizeof(int32_t));
            rdmaHead = rdmaRecvHeadTailTensor_.GetValue(currentServerId * RING_BUFFER_HEAD_TAIL / sizeof(int32_t) + 1 * 8);
            printf("before RANK%d AIVID%d i=%d rdmaTail%d rdmaHead%d size%d currentServerId%d\n",
            rankId_, aivId_, i, rdmaTail, rdmaHead, EXP_TOKEN_COUNT_FLAG_CNT +
            currentServerId * SERVER_SIZE_ON_WIN + rdmaHead * tokenStructLen_, currentServerId);
            // 当前环为空的时候直接跳到下一个环，不在空环中遍历token
            if (rdmaHead % rdmaItemNum == rdmaTail % rdmaItemNum) {
                printf("continue RANK%d AIVID%d i=%d\n", rankId_, aivId_, i);
                i = currentServerId + 1 < serverNum ? (currentServerId + 1) * rdmaItemNum : 0;
                continue;
            }
            uint32_t destOffset = EXP_TOKEN_COUNT_FLAG_CNT + currentServerId * SERVER_SIZE_ON_WIN +
                rdmaHead * tokenStructLen_;
            printf("Rdma2HCCS RANK%d AIVID %d i=%d rdmaTail:%d rdmaHead%d\n", rankId_, aivId_, i, rdmaTail, rdmaHead);
            DataCopyPad(tokenStructInRdmaTensor_,
            rdmaRecvRingU8Tensor_[EXP_TOKEN_COUNT_FLAG_CNT + currentServerId * SERVER_SIZE_ON_WIN +
                rdmaHead * tokenStructLen_],
            tokenStructParams, tokenStructPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t localTokenIdx = tokenStructInRdmaTensor_.ReinterpretCast<uint32_t>().
                GetValue(cntOffsetInStruct_/sizeof(uint32_t));
            printf("RANK%d AIVID%d 809 addr%p localTokenIdx=%d currentServerId%d rdmaRecvRingU8Tensor_:%d\n",
            rankId_, aivId_, destOffset + cntOffsetInStruct_ + windowInGM_,
            localTokenIdx, currentServerId, rdmaRecvRingU8Tensor_(destOffset + cntOffsetInStruct_));
            uint32_t preRecvTokenCnt = 0;
            for (int j = 0; j < currentServerId; j++) {
                preRecvTokenCnt += rdmaRecvRingU32Tensor_.GetValue(j * SERVER_SIZE_ON_WIN / sizeof(uint32_t) +
                                1 * UB_32B_ALIGN / sizeof(uint32_t));
            }
            
            uint32_t globalTokenIdx = preRecvTokenCnt + localTokenIdx;
            printf("Rdma2HCCS RANK%d AIVID %d 822 globalTokenIdx:%d preRecvTokenCnt:%d localTokenIdx:%d\n",
            rankId_, aivId_, globalTokenIdx, preRecvTokenCnt, localTokenIdx);
            tokenStructInRdmaTensor_.ReinterpretCast<uint32_t>().
                SetValue(cntOffsetInStruct_/sizeof(uint32_t), globalTokenIdx);
            for (int j = 0; j < axisK_; j++) {
                int32_t dstExpert = tokenStructInRdmaTensor_.ReinterpretCast<int32_t>().
                    GetValue(expOffsetInStruct_/sizeof(int32_t) + j);
                printf("Rdma2HCCS RANK%d COREIDX%d topk%d = %d expertIdxStart:%d ,expertIdxEnd%d\n",
                rankId_, aivId_, j, dstExpert, expertIdxStart, expertIdxEnd);
                if (dstExpert < expertIdxStart || dstExpert >= expertIdxEnd) {
                    continue;
                }
                uint32_t dstRank = dstExpert / localMoeExpertNum_;
                uint32_t localDstRank = dstRank % SERVER_RANK_SIZE;
                // 每个核只处理发往本核对应的rank的数据
                if (localDstRank != localWorkCoreId) {
                    continue;
                }
                GlobalTensor<uint8_t> dstRankRecvRingU8Tensor;
                dstRankRecvRingU8Tensor.SetGlobalBuffer((__gm__ uint8_t *)(hccl_.GetWindowsInAddr(dstRank) +
                                        totalWinSize_ + IPC_DATA_OFFSET + localRankId * RANK_SIZE_ON_IPC));
                GlobalTensor<uint32_t> globalHccsHeadTailTensor;
                globalHccsHeadTailTensor.SetGlobalBuffer((__gm__ uint32_t *)
                    reinterpret_cast<uint64_t>(hccsHeadTailGM[localDstRank] +
                    EACH_HCCS_RING_BUFFER_HEAD_TAIL * localRankId));
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                DataCopy<uint32_t>(localHccsHeadTailTensor_, globalHccsHeadTailTensor,
                        EACH_HCCS_RING_BUFFER_HEAD_TAIL / sizeof(uint32_t));
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                uint32_t hcclTail = localHccsHeadTailTensor_.GetValue(0 * UB_32B_ALIGN / sizeof(uint32_t));
                uint32_t hcclHead = localHccsHeadTailTensor_.GetValue(1 * UB_32B_ALIGN / sizeof(uint32_t));
                printf("wait hccsBuffer RANK%d AIVID%d topk:%d hcclTail:%d hcclHead:%d localHccsHeadTailAddr:%p\n",
                    rankId_, aivId_, j, hcclTail, hcclHead,
                    (__gm__ uint32_t *)
                    reinterpret_cast<uint64_t>(hccsHeadTailGM[localDstRank] +
                    EACH_HCCS_RING_BUFFER_HEAD_TAIL * localRankId));
                while (hcclHead == (hcclTail + 1) % hccsItemNum) {
                    DataCopy<uint32_t>(localHccsHeadTailTensor_, globalHccsHeadTailTensor,
                        EACH_HCCS_RING_BUFFER_HEAD_TAIL / sizeof(uint32_t));
                    SyncFunc<AscendC::HardEvent::MTE2_S>();
                    hcclHead = localHccsHeadTailTensor_.GetValue(1 * UB_32B_ALIGN / sizeof(uint32_t));
                }
                printf("after hccsBuffer RANK%d AIVID%d topk:%d hcclTail:%d hcclHead:%d localDstRank:%d localRankId:%d dstRank %d localMoeExpertNum_ %d\n",
                    rankId_, aivId_, j, hcclTail, hcclHead, localDstRank, localRankId, dstRank, localMoeExpertNum_);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                // 每张卡上面分配localranksize块共享内存（物理连续，逻辑离散），分别存储来自于不同rank的token信息
                DataCopyPad(dstRankRecvRingU8Tensor[tokenStructLen_ * hcclTail],
                            tokenStructInRdmaTensor_, tokenStructParams);
                printf("RDMA2HCCS RANK %d AIVID %d servertokenId(ub) %d servertokenId(gm) %d dstRankRecvRingU8Tensor %p\n",
                    rankId_, aivId_,
                    tokenStructInRdmaTensor_(cntOffsetInStruct_),
                    dstRankRecvRingU8Tensor(tokenStructLen_ * hcclTail + cntOffsetInStruct_),
                    (hccl_.GetWindowsInAddr(dstRank) +
                    totalWinSize_ + IPC_DATA_OFFSET + localRankId * RANK_SIZE_ON_IPC));
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                hcclTail = (hcclTail + 1) % hccsItemNum;
                localHccsHeadTailTensor_.SetValue(0 * UB_32B_ALIGN / sizeof(uint32_t), hcclTail);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                printf("RANK%d AIVID%d Rdma2HCCS 1047 localHccsHeadTailTensor_:%d hcclTail:%d\n",
                    rankId_, localDstRank, localHccsHeadTailTensor_(0), hcclTail);
                // 每张卡一段内存记录用于接收来自各个卡环形buffer的headtail，localranksize个headtail与localranksize个共享内存对应
                DataCopy<uint32_t>(globalHccsHeadTailTensor, localHccsHeadTailTensor_, UB_32B_ALIGN / sizeof(uint32_t));
                printf("RANK%d AIVID%d Rdma2HCCS 1052 globalHccsHeadTailTensor:%d hccsHeadTailTensorAddr:%p hcclTail:%d\n",
                rankId_, localDstRank, globalHccsHeadTailTensor(0),
                (__gm__ uint32_t *)
                    reinterpret_cast<uint64_t>(hccsHeadTailGM[localDstRank] +
                    EACH_HCCS_RING_BUFFER_HEAD_TAIL * localRankId),
                hcclTail);
                break; // 每个核负责一个目的卡，此处为了去重做的 break
            }
            processedTokenNum++;
            if (localWorkCoreId == 0) {
                rdmaHead = (rdmaHead + 1) % rdmaItemNum;
                rdmaRecvHeadTailTensor_.SetValue(currentServerId * 2 * UB_32B_ALIGN / sizeof(int32_t) + 1 * 8, rdmaHead);
                AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                        AscendC::DcciDst::CACHELINE_OUT>(rdmaRecvHeadTailTensor_[currentServerId * 2 * UB_32B_ALIGN / sizeof(int32_t) + 1 * 8]);
            }
            printf("before syncall RANK%d AIVID %d i=%d rdmaTail:%d rdmaHead%d\n", rankId_, aivId_, i, rdmaTail, rdmaHead);
            SyncAll(syncCoreGMTensor_, localSyncCoreTensor_, SERVER_RANK_SIZE);
            printf("RANK %d AIVID %d processed %d tokenGlobalCnt %d\n", rankId_, aivId_, processedTokenNum, tokenGlobalCnt);
            if (processedTokenNum >= tokenGlobalCnt) {
                hccsBufferStatusTensor_.SetValue(0, HCCS_SEND_END);
                break;
            }
        }
    }
    printflag(after while loop);
    // while rdma_head < rdma_tail:
    //     wait_hccs_space()
    //     dma_async(rdma_buf[rdma_head], hccs_buf[hccs_tail])
    //     rdma_head++
    //     hccs_tail++
}

// 由FORWARDER_COORDINATOR执行，负责将rdmaRecvRingU8Tensor这个环形buffer中已发送到hccsRecvRingU8Tensor的部分进行回收
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::CreditRecycle()
{
    if (aivRole_ != FORWARDER_COORDINATOR) {
        return;
    }
    // if rdma_head - last_reported >= CREDIT_BATCH:
    //     rdma_atomic_add(remote_head, CREDIT_BATCH)
    //     last_reported += CREDIT_BATCH
}

// 由HCCS_RECEIVER执行，负责将hccsRecvRingU8Tensor的数据发送到最终输出
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::HCCS2Out()
{
    if (aivRole_ != HCCS_RECEIVER) {
        return;
    }
    // 每个核负责一个rank的环形buffer
    uint32_t senderNum = aivNum_ / 4 - 1;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    uint32_t tokenNumPerCore = hccsItemNum / senderNum;
    uint32_t remaind = hccsItemNum % senderNum;
    uint32_t localStartWorkerCoreIdx = (HCCS_RECEIVER - 1) * senderNum;
    uint32_t localWorkerCoreIdx = aivId_ - localStartWorkerCoreIdx;
    uint32_t expertIdxStart = localMoeExpertNum_ * rankId_;
    uint32_t expertIdxEnd = expertIdxStart + localMoeExpertNum_;
    if (localWorkerCoreIdx >= SERVER_RANK_SIZE) {
        return;
    }
    LocalTensor<float> weightTmp = weightBuf_.Get<float>();
    uint32_t processedTokens = 0;
    DataCopyExtParams tokenStructParams{1, static_cast<uint32_t>(tokenStructLen_), 0, 0, 0};
    DataCopyExtParams tokenParams{1, static_cast<uint32_t>(tokenLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> tokenStructPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams weightParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> weightExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams scalesParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> scalesExtParams{false, 0U, 0U, 0U};
    DataCopyParams hccsHesdTailParams{1, EACH_HCCS_RING_BUFFER_HEAD_TAIL / sizeof(uint32_t), 0, 0};
    GlobalTensor<uint32_t> hccsHeadTailTensor;
    hccsHeadTailTensor.SetGlobalBuffer((__gm__ uint32_t *)(reinterpret_cast<uint64_t>
    (hccsHeadTailGM[localRankId] + EACH_HCCS_RING_BUFFER_HEAD_TAIL * localWorkerCoreIdx)));
    DataCopy(localHccsHeadTailTensor_, hccsHeadTailTensor, EACH_HCCS_RING_BUFFER_HEAD_TAIL / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t hccsHead = localHccsHeadTailTensor_.GetValue(1 * UB_32B_ALIGN / sizeof(uint32_t));
    uint32_t hccsTail = localHccsHeadTailTensor_.GetValue(0 * UB_32B_ALIGN / sizeof(uint32_t));
    uint32_t flag = hccsBufferStatusTensor_.GetValue(0);
    hccsRecvRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_) + totalWinSize_ + IPC_DATA_OFFSET + localWorkerCoreIdx * RANK_SIZE_ON_IPC)));
    
    printf("RANK%d AIVID%d HCCS2Out hccsRecvRingU8Tensor_:%p hccsHeadTailTensor:%p\n",
            rankId_, localWorkerCoreIdx,
            (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_) + totalWinSize_ + IPC_DATA_OFFSET + localRankId * RANK_SIZE_ON_IPC)),
            (__gm__ uint32_t *)(reinterpret_cast<uint64_t>
            (hccsHeadTailGM[localRankId] + EACH_HCCS_RING_BUFFER_HEAD_TAIL * localWorkerCoreIdx)));
    uint32_t count = 0;
    while (hccsHead != hccsTail % hccsItemNum || flag != HCCS_SEND_END) {
        if (hccsHead == hccsTail % hccsItemNum) {
            DataCopy(localHccsHeadTailTensor_, hccsHeadTailTensor, EACH_HCCS_RING_BUFFER_HEAD_TAIL / sizeof(uint32_t));
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            hccsTail = localHccsHeadTailTensor_.GetValue(0 * UB_32B_ALIGN / sizeof(uint32_t));
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            flag = hccsBufferStatusTensor_.GetValue(0);
            if (count == 100000) {
                printf("HCCS2Out RANK%d AIVID%d hccsHead:%d hccsTail%d flag%d\n", rankId_, aivId_, hccsHead, hccsTail, flag);
                //AscendC::DumpTensor(hccsRecvRingU8Tensor_[expOffsetInStruct_], 333, 32);
            }
            count++;
            continue;
        }
        DataCopyPad(tokenStructInHccsTensor_, hccsRecvRingU8Tensor_[tokenStructLen_ * hccsHead],
                    tokenStructParams, tokenStructPadParams);
        printf("RANK %d AIVID %d HCCS2Out 1186 hccsHead %d hccsTail %d servertokenId(ub) %d servertokenId(gm) %d hccsRecvRingU8Tensor_ %p\n",
                    rankId_, aivId_, hccsHead, hccsTail,
                    tokenStructInHccsTensor_(cntOffsetInStruct_),
                    hccsRecvRingU8Tensor_(tokenStructLen_ * hccsHead + cntOffsetInStruct_),
                    hccl_.GetWindowsInAddr(rankId_) + totalWinSize_ + IPC_DATA_OFFSET +
                    localWorkerCoreIdx * RANK_SIZE_ON_IPC);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        LocalTensor<uint32_t> tokenIdxInStructTensor = tokenStructInHccsTensor_[cntOffsetInStruct_].
                                                        ReinterpretCast<uint32_t>();
        LocalTensor<float> weightTensor = tokenStructInHccsTensor_[weightOffsetInStruct_].ReinterpretCast<float>();
        LocalTensor<ExpandXOutType> tokenOutTensor = tokenStructInHccsTensor_.ReinterpretCast<ExpandXOutType>();
        uint32_t tokenIdx = tokenIdxInStructTensor.GetValue(0);
        LocalTensor<int32_t> topkIdxTensor = tokenStructInHccsTensor_[expOffsetInStruct_].ReinterpretCast<int32_t>();
        printf("HCCS2Out RANK%d AIVID%d tokenIdx(%d)\n", rankId_, aivId_, tokenIdx);
        uint32_t dstOffset = 0;
        for (int j = 0; j < axisK_; j++) {
            uint32_t dstExpert = topkIdxTensor.GetValue(j);
            if (dstExpert < expertIdxStart || dstExpert >= expertIdxEnd) {
                continue;
            }
            dstOffset = tokenIdxPerExpertGMTensor_.GetValue(tokenIdx * axisK_ + j);
            weightTmp(0) = weightTensor.GetValue(j);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            pipe_barrier(PIPE_ALL);
            DataCopyPad(weightsOutGt[dstOffset], weightTmp, weightParams);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(expandXOutGMTensor_[dstOffset * axisH_], tokenOutTensor, tokenParams);
            // dynamic scales to output
            printf("RANK %d AIVID %d dstOffset %d weightTmp %f j %d tokenId %d \n",
                rankId_, aivId_, dstOffset, weightTmp(0), j, tokenIdx);
            if constexpr (DynamicQuant) {
                LocalTensor<float> quantTempUB =
                    tokenStructInHccsTensor_[scaleOffsetInStruct_].ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[dstOffset], quantTempUB, scalesParams);
            }
        }
        hccsHead = (hccsHead + 1) % hccsItemNum;
        localHccsHeadTailTensor_.SetValue(1 * UB_32B_ALIGN / sizeof(uint32_t), hccsHead);
        DataCopy(hccsHeadTailTensor[1 * UB_32B_ALIGN / sizeof(uint32_t)],
                localHccsHeadTailTensor_[1 * UB_32B_ALIGN / sizeof(uint32_t)],
                UB_32B_ALIGN / sizeof(uint32_t));
        printf("HCCS2OUT RANK %d AIVID %d hccsTail %d hccsHead %d flag %d\n", rankId_, aivId_, hccsHeadTailTensor(0),
                hccsHeadTailTensor(1 * UB_32B_ALIGN / sizeof(uint32_t)), flag);
    }
    // while hccs_head < hccs_tail:
    //     dma_async(hccs_buf[hccs_head], recv_x[offset])
    //     hccs_head++
}

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline uint64_t
MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::MergeMagicWithValue(uint64_t magic, uint64_t value)
{
    return (magic * 2ULL + value);
}

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        WorkBeforePipeline();
        PrepareRdmaSend();
        TriggerRdmaSend();
        Rdma2HCCS();
        // CreditRecycle();
        HCCS2Out();

        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
