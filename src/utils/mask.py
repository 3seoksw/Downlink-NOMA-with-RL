import torch


def get_mask(state, num_features, num_users, num_channels, device):
    N = num_users
    K = num_channels
    batch_size = state.shape[0]

    mask = torch.zeros(batch_size, N * K, dtype=torch.bool, device=device)

    state_status = state[:, :, -1]
    if num_features == 3:
        indices = torch.nonzero(state_status)
    elif num_features == 1 or num_features == 2:
        indices = (state_status == 0).nonzero()
    else:
        raise KeyError()

    batch_indices = indices[:, 0]
    state_indices = indices[:, 1]
    channel_indices = state_indices // N
    user_indices = state_indices % N

    # User masking
    for batch_idx, user_idx, channel_idx in zip(
        batch_indices, user_indices, channel_indices
    ):
        channel_indices = torch.arange(K, device=device)
        state_indices = user_idx + channel_indices * N
        mask[batch_idx, state_indices] = True

    # Channel masking
    if num_features == 1 or num_features == 2:
        state_status = ~state_status.bool()
    state_matrix = state_status.view(batch_size, K, N, -1)
    assigned_counts = state_matrix.sum(dim=-1).bool().sum(dim=-1)
    full_channels = (assigned_counts >= 2).nonzero(as_tuple=True)
    for batch_idx, channel_idx in zip(*full_channels):
        state_indices = torch.arange(N, device=device) + channel_idx * N
        mask[batch_idx, state_indices] = True

    return mask
