#!/bin/bash

# ==================== ??? ====================
TEST_CMD="bash run_test_internode.sh"
ROUNDS=20
LOG_FILE="perf_output.log"
# ===============================================

rm -f "$LOG_FILE"

echo "?? ?? $ROUNDS ?????..."
echo "??: $TEST_CMD"
echo "----------------------------------------"

for ((i=1; i<=ROUNDS; i++)); do
    echo ">>> Round $i <<<" >> "$LOG_FILE"
    echo -n "? $i ?... "

    if output=$($TEST_CMD 2>&1); then
        echo "? ??"
        echo "$output" >> "$LOG_FILE"
    else
        echo "? ??"
        echo "[ERROR] Round $i failed" >> "$LOG_FILE"
	echo "$output" > error.log
    fi
done

echo ""
echo "?? ????????????..."

# ==================== awk ?? ====================
awk '
# ---------------- Dispatch ----------------
/\[tuning\].*Dispatch/ {
    if (match($0, /[0-9.]+ GB\/s \(HCCS\)/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/ GB\/s \(HCCS\)/, "", val)
        d_hccs_sum += val; d_hccs_cnt++
    }

    if (match($0, /[0-9.]+ GB\/s \(RDMA\)/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/ GB\/s \(RDMA\)/, "", val)
        d_rdma_sum += val; d_rdma_cnt++
    }

    if (match($0, /avg_t: [0-9.]+ us/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/avg_t: /, "", val)
        sub(/ us/, "", val)
        d_time_sum += val; d_time_cnt++
    }
}

# ---------------- Combine ----------------
/\[tuning\].*Combine/ {
    if (match($0, /[0-9.]+ GB\/s \(HCCS\)/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/ GB\/s \(HCCS\)/, "", val)
        c_hccs_sum += val; c_hccs_cnt++
    }

    if (match($0, /[0-9.]+ GB\/s \(RDMA\)/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/ GB\/s \(RDMA\)/, "", val)
        c_rdma_sum += val; c_rdma_cnt++
    }

    if (match($0, /avg_t: [0-9.]+ us/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/avg_t: /, "", val)
        sub(/ us/, "", val)
        c_time_sum += val; c_time_cnt++
    }
}

# ---------------- Kernel ----------------
/\[layout\].*Kernel performance:/ {
    if (match($0, /Kernel performance: [0-9.]+ ms/)) {
        val = substr($0, RSTART, RLENGTH)
        sub(/Kernel performance: /, "", val)
        sub(/ ms/, "", val)
        k_sum += val * 1000
        k_cnt++
    }
}

END {
    print ""
    print "?? ?????? (???)"
    print "----------------------------------------"

    if (d_time_cnt) {
        printf "Dispatch avg_t    : %.2f us (n=%d)\n", d_time_sum/d_time_cnt, d_time_cnt
        printf "Dispatch HCCS BW  : %.2f GB/s\n", d_hccs_sum/d_hccs_cnt
        printf "Dispatch RDMA BW  : %.2f GB/s\n", d_rdma_sum/d_rdma_cnt
    } else {
        print "Dispatch: ? ???"
    }

    print ""

    if (c_time_cnt) {
        printf "Combine  avg_t    : %.2f us (n=%d)\n", c_time_sum/c_time_cnt, c_time_cnt
        printf "Combine  HCCS BW  : %.2f GB/s\n", c_hccs_sum/c_hccs_cnt
        printf "Combine  RDMA BW  : %.2f GB/s\n", c_rdma_sum/c_rdma_cnt
    } else {
        print "Combine: ? ???"
    }

    print ""

    if (k_cnt) {
        printf "Kernel time avg   : %.2f us (n=%d)\n", k_sum/k_cnt, k_cnt
    } else {
        print "Kernel: ? ???"
    }
}
' "$LOG_FILE"

echo ""
echo "?? ????????: $LOG_FILE"
