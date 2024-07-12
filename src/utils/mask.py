import torch


def get_mask(state, num_features, num_users, num_channels, device):
    N = num_users
    K = num_channels
    batch_size = state.shape[0]

    mask = torch.zeros(batch_size, N * K, dtype=torch.bool, device=device)

    state_status = state[:, :, -1]
    if num_features == 3:
        indices = torch.nonzero(state_status, as_tuple=True)
    elif num_features == 1 or num_features == 2:
        indices = torch.nonzero(state_status == 0, as_tuple=True)
    else:
        raise KeyError()

    batch_indices, state_indices = indices
    user_indices = state_indices % N

    # User masking
    user_state_indices = (
        user_indices.unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0) * N
    )
    mask[batch_indices.unsqueeze(1), user_state_indices] = True

    # Channel masking
    if num_features == 1 or num_features == 2:
        state_status = ~state_status.bool()
    state_matrix = state_status.view(batch_size, K, N)
    assigned_counts = state_matrix.sum(dim=-1) >= 2
    full_channels = assigned_counts.nonzero(as_tuple=True)

    if len(full_channels[0]) > 0:
        full_batch_indices, full_channel_indices = full_channels
        full_state_indices = (
            torch.arange(N, device=device).unsqueeze(0)
            + full_channel_indices.unsqueeze(1) * N
        )
        mask[full_batch_indices.unsqueeze(1), full_state_indices] = True

    return mask
