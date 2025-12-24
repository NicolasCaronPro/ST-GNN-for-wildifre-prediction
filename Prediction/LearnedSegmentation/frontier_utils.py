import numpy as np

def ids_to_boundary_mask(L: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    L: (H, W) int array
    returns B: (H, W) uint8, 1 on boundary, 0 otherwise
    """
    H, W = L.shape
    B = np.zeros((H, W), dtype=np.uint8)

    diff_r = np.zeros_like(B)
    diff_r[:, :-1] = (L[:, :-1] != L[:, 1:]).astype(np.uint8)
    diff_d = np.zeros_like(B)
    diff_d[:-1, :] = (L[:-1, :] != L[1:, :]).astype(np.uint8)

    B |= diff_r
    B |= diff_d

    B[:, 1:] |= diff_r[:, :-1]
    B[1:, :] |= diff_d[:-1, :]

    if connectivity == 8:
        diff_dr = np.zeros_like(B)
        diff_dr[:-1, :-1] = (L[:-1, :-1] != L[1:, 1:]).astype(np.uint8)
        diff_dl = np.zeros_like(B)
        diff_dl[:-1, 1:] = (L[:-1, 1:] != L[1:, :-1]).astype(np.uint8)

        B |= diff_dr
        B |= diff_dl

        B[1:, 1:] |= diff_dr[:-1, :-1]
        B[1:, :-1] |= diff_dl[:-1, 1:]

    return B

def connected_components_8(binary: np.ndarray) -> np.ndarray:
    """
    Simple BFS connected components (8-connectivity).
    """
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current = 0

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    for y in range(H):
        for x in range(W):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels
