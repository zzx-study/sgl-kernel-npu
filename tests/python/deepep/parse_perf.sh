#!/bin/bash

# ==================== 配置区 ====================
TEST_CMD="bash run_test_internode.sh"    # ← 替换为你的实际测试命令
ROUNDS=5                     # 测试轮数
LOG_FILE="perf_output.log"   # 临时日志文件
# ===============================================

# 清理旧日志
rm "$LOG_FILE"

echo "������ 开始  $ROUNDS 轮性能测试..."
echo "命令:  $TEST_CMD"
echo "----------------------------------------"

# 运行多轮测试，追加输出到日志
for ((i=1; i<=ROUNDS; i++)); do
    echo ">>> Round  $i <<<" >> "$LOG_FILE"
    echo -n "第  $i 轮... "
    
    # 执行命令，同时输出到终端和日志（可选）
    if output=$($TEST_CMD 2>&1); then
        echo "✅ 成功"
        echo " $output" >> "$LOG_FILE"
    else
        echo "❌ 失败 (exit code:  $?)"
        echo "[ERROR] Round  $i failed" >> "$LOG_FILE"
    fi
done

echo ""
echo "������ 正在解析日志并计算统计量..."

dispatch=()
combine=()
kernel=()

while IFS= read -r line; do
    if [[ "$line" =~ \[tuning\].*Dispatch.*avg_t:\ ([0-9.]+)\ us ]]; then
        dispatch+=("${BASH_REMATCH[1]}")
    elif [[ "$line" =~ \[tuning\].*Combine.*avg_t:\ ([0-9.]+)\ us ]]; then
        combine+=("${BASH_REMATCH[1]}")
    elif [[ "$line" =~ \[layout\].*Kernel\ performance:\ ([0-9.]+)\ ms ]]; then
        us=$(awk "BEGIN{print ${BASH_REMATCH[1]} * 1000}")
        kernel+=("$us")
    fi
done < "$LOG_FILE"

avg() {
    local sum=0
    for v in "$@"; do
        sum=$(awk "BEGIN{print $sum + $v}")
    done
    awk "BEGIN{print $sum / $#}"
}

echo "Dispatch avg_t: $(avg "${dispatch[@]}") us"
echo "Combine  avg_t: $(avg "${combine[@]}") us"
echo "Kernel time : $(avg "${kernel[@]}") us"
# 可选：保留日志供复查
echo ""
echo "������ 详细日志已保存至:  $LOG_FILE"
