import os

import torch
import torch.distributed as dist
import torch_npu

from .utils import EventOverlap

COMM_STREAM = None


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        a2a_out = torch.empty_like(input_)
    else:
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )

    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.npu.current_device()
    )
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    return output


def alltoall_get_dispatch_layout(buffer, topk_idx, num_experts):
    group = buffer.group
    group_size = buffer.group_size
    num_local_experts = num_experts // group_size
    ep_rank = buffer.rank
    device = topk_idx.device

    num_local_tokens_per_expert = torch.histc(
        topk_idx, bins=num_experts, min=0, max=num_experts
    )

    input_splits = (
        num_local_tokens_per_expert.reshape(group_size, num_local_experts)
        .sum(axis=1)
        .cpu()
        .numpy()
        .tolist()
    )

    num_global_tokens_per_expert = _gather_along_first_dim(
        num_local_tokens_per_expert, group
    ).reshape(group_size, num_experts)

    local_expert_indices_offset = ep_rank * num_local_experts
    local_expert_indices = [
        local_expert_indices_offset + i for i in range(num_local_experts)
    ]

    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, local_expert_indices[0] : local_expert_indices[-1] + 1
    ]

    output_splits = (
        num_global_tokens_per_local_expert.sum(axis=-1).cpu().numpy().tolist()
    )

    num_tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

    expert_ids_per_ep_rank = (
        torch.arange(
            num_experts,
            dtype=torch.int32,
            device=device,
        )
        % num_local_experts
    )

    num_global_tokens_per_local_expert_ravel = (
        num_global_tokens_per_local_expert.ravel()
    )
    if num_local_experts > 1:
        global_tokens_indices = torch.repeat_interleave(
            expert_ids_per_ep_rank,
            num_global_tokens_per_local_expert_ravel,
        )
    else:
        torch.npu.synchronize()
        global_tokens_indices = None

    layout = {
        "num_local_experts": num_local_experts,
        "input_splits": input_splits,
        "output_splits": output_splits,
        "num_global_tokens_per_local_expert": num_global_tokens_per_local_expert,
        "global_tokens_indices": global_tokens_indices,
    }
    buffer._alltoall_layout = layout

    num_tokens_per_rank = num_local_tokens_per_expert.reshape(
        group_size, num_local_experts
    ).sum(axis=1)
    is_token_in_rank = torch.zeros(
        (topk_idx.size(0), group_size), dtype=torch.bool, device=device
    )

    return (
        num_tokens_per_rank,
        None,
        num_tokens_per_expert,
        is_token_in_rank,
        EventOverlap(),
    )


def alltoall_dispatch(buffer, x, topk_idx, topk_weights):
    layout = buffer._alltoall_layout
    num_local_experts = layout["num_local_experts"]
    input_splits = layout["input_splits"]
    output_splits = layout["output_splits"]
    num_global_tokens_per_local_expert = layout["num_global_tokens_per_local_expert"]
    global_tokens_indices = layout["global_tokens_indices"]

    hidden_shape = x.shape
    x = x.view(-1, hidden_shape[-1])

    permutated_tokens, reversed_local_mapping = torch_npu.npu_moe_token_permute(
        tokens=x,
        indices=topk_idx,
        num_out_tokens=topk_idx.numel(),
    )

    input_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
    if input_quant:
        permutated_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
            permutated_tokens
        )
        _, dynamic_scale_after_all2all, scale_handle = async_all_to_all(
            dynamic_scale, output_splits, input_splits, buffer.group
        )
        scale_handle.wait()
        dynamic_scale.untyped_storage().resize_(0)

    _, global_input_tokens, handle = async_all_to_all(
        permutated_tokens,
        output_splits,
        input_splits,
        buffer.group,
    )
    handle.wait()
    permutated_tokens.untyped_storage().resize_(0)

    if num_local_experts > 1:
        if input_quant:
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1), global_tokens_indices
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        dispatch_out, reversed_global_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_tokens_indices
        )
    else:
        dispatch_out = global_input_tokens
        reversed_global_mapping = None

    num_recv_tokens_per_expert_list = (
        num_global_tokens_per_local_expert.sum(axis=0).cpu().numpy().tolist()
    )

    combine_handle = {
        "input_splits": input_splits,
        "output_splits": output_splits,
        "topk_weights": topk_weights,
        "reversed_local_mapping": reversed_local_mapping,
        "reversed_global_mapping": reversed_global_mapping,
        "hidden_shape": hidden_shape,
        "hidden_shape_before_permute": x.shape,
        "num_local_experts": num_local_experts,
    }

    recv_x = (
        (dispatch_out, dynamic_scale_after_all2all) if input_quant else dispatch_out
    )

    return (
        recv_x,
        None,
        None,
        num_recv_tokens_per_expert_list,
        combine_handle,
        EventOverlap(),
    )


def alltoall_combine(buffer, x, handle):
    input_splits = handle["input_splits"]
    output_splits = handle["output_splits"]
    topk_weights = handle["topk_weights"]
    reversed_local_mapping = handle["reversed_local_mapping"]
    reversed_global_mapping = handle["reversed_global_mapping"]
    hidden_shape = handle["hidden_shape"]
    hidden_shape_before_permute = handle["hidden_shape_before_permute"]
    num_local_experts = handle["num_local_experts"]

    if x.shape[0] > 0 and num_local_experts > 1 and reversed_global_mapping is not None:
        x = torch_npu.npu_moe_token_unpermute(x, reversed_global_mapping)

    _, local_tokens, a2a_handle = async_all_to_all(
        x,
        input_splits,
        output_splits,
        buffer.group,
    )
    a2a_handle.wait()
    x.untyped_storage().resize_(0)

    output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=local_tokens,
        sorted_indices=reversed_local_mapping.to(torch.int32),
        probs=topk_weights,
        restore_shape=hidden_shape_before_permute,
    )
    output = output.view(hidden_shape)

    return output, None, EventOverlap()


def alltoall_low_latency_dispatch(
    buffer,
    x,
    topk_idx,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    cumulative_local_expert_recv_stats=None,
    use_fp8=True,
    round_scale=False,
    use_ue8m0=False,
    async_finish=False,
    return_recv_hook=False,
):
    group = buffer.group
    group_size = buffer.group_size
    num_local_experts = num_experts // group_size
    device = x.device
    hidden = x.size(1)
    aligned_num_tokens = num_max_dispatch_tokens_per_rank
    num_tokens = x.size(0)
    x_padding = torch.zeros(
        (aligned_num_tokens, hidden),
        dtype=x.dtype,
        device=x.device,
    )
    topk_padding = torch.zeros(
        aligned_num_tokens, topk_idx.size(1),
        dtype=x.dtype,
        device=x.device,
    )
    topk_padding[:num_tokens].copy_(topk_idx)
    x_padding[:num_tokens].copy_(x)

    topk_idx_int = topk_padding.to(torch.int32)
    expert_capacity = aligned_num_tokens
    (expanded_x, expanded_row_idx, _, _) = (
        torch_npu.npu_moe_init_routing_v2(
            x_padding,
            topk_idx_int,
            quant_mode=-1,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=1,
            expert_capacity=expert_capacity,
            active_expert_range=[0, num_experts],
        )
    )

    expanded_x_2d = expanded_x.reshape(num_experts * expert_capacity, hidden)
    chunk_size = num_local_experts * expert_capacity

    if use_fp8:
        expanded_x_int8, expanded_x_scales = torch_npu.npu_dynamic_quant(expanded_x_2d)
        expanded_x_scales_bf16 = expanded_x_scales.to(torch.bfloat16).unsqueeze(-1)
        combined_hidden = hidden + 1
        combined_send = torch.cat(
            [
                expanded_x_int8.to(torch.bfloat16),
                expanded_x_scales_bf16,
            ],
            dim=1,
        )

        input_list = [
            combined_send[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(
                chunk_size, combined_hidden, dtype=torch.bfloat16, device=device
            )
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_combined_raw = torch.cat(output_list, dim=0)

        recv_combined = recv_combined_raw.reshape(
            group_size, num_local_experts, expert_capacity, combined_hidden
        )
        recv_combined = recv_combined.permute(1, 0, 2, 3).contiguous()
        recv_combined = recv_combined.reshape(
            num_local_experts * group_size * expert_capacity, combined_hidden
        )

        recv_x_int8 = recv_combined[:, :hidden].to(torch.int8)
        recv_x_scales = (
            recv_combined[:, hidden : hidden + 1].squeeze(-1).to(torch.float32)
        )
        recv_x_out = (recv_x_int8, recv_x_scales)
    else:
        input_list = [
            expanded_x_2d[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=expanded_x_2d.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_x_raw = torch.cat(output_list, dim=0)

        recv_all = recv_x_raw.reshape(
            group_size, num_local_experts, expert_capacity, hidden
        )
        recv_all = recv_all.permute(1, 0, 2, 3).contiguous()
        recv_x = recv_all.reshape(
            num_local_experts * group_size * expert_capacity, hidden
        )
        recv_x_out = recv_x

    packed_recv_count = torch.full(
        (num_local_experts,), expert_capacity, dtype=torch.int64, device=x.device,
    )

    handle_tuple = (
        expanded_row_idx,
        expert_capacity,
        hidden,
        num_tokens,
        num_local_experts,
        group_size,
        packed_recv_count,
    )

    return (
        recv_x_out,
        packed_recv_count,
        handle_tuple,
        EventOverlap(),
        lambda: None,
    )


def alltoall_low_latency_combine(
    buffer,
    x,
    topk_idx,
    topk_weights,
    handle,
    zero_copy=False,
    async_finish=False,
    return_recv_hook=False,
    out=None,
):
    expanded_row_idx = handle[0]
    expert_capacity = handle[1]
    hidden = handle[2]
    num_tokens = handle[3]
    num_local_experts = handle[4]
    group_size = handle[5]

    device = x.device
    group = buffer.group

    x_reordered = x.reshape(num_local_experts, group_size, expert_capacity, hidden)
    x_reordered = x_reordered.permute(1, 0, 2, 3).contiguous()
    x_reordered = x_reordered.reshape(
        group_size * num_local_experts * expert_capacity, hidden
    )

    chunk_size = num_local_experts * expert_capacity
    input_list = [
        x_reordered[r * chunk_size : (r + 1) * chunk_size].contiguous()
        for r in range(group_size)
    ]
    output_list = [
        torch.empty(chunk_size, hidden, dtype=x.dtype, device=device)
        for r in range(group_size)
    ]
    dist.all_to_all(output_list, input_list, group=group)
    recv_all_raw = torch.cat(output_list, dim=0)

    recv_all_raw = recv_all_raw.reshape(
        group_size * num_local_experts, expert_capacity, hidden
    )
    topk_weights_padding = torch.zeros(
        expert_capacity, topk_weights.size(1),
        dtype=x.dtype,
        device=x.device,
    )
    topk_weights_padding[:num_tokens].copy_(topk_weights)
    output = torch_npu.npu_moe_finalize_routing(
        expanded_permuted_rows=recv_all_raw,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights_padding,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=None,
        drop_pad_mode=3,
    )
    output = output[:num_tokens,:]

    return output, EventOverlap(), lambda: None
