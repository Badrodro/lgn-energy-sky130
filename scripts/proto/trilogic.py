import torch
def tri_lut(lo_a, hi_a, lo_b, hi_b, T):
    # each signal: (can_be_0, can_be_1) encoded as lo (min value) / hi (max value) in {0,1}
    lo = torch.ones_like(lo_a); hi = torch.zeros_like(lo_a)
    for av in (0, 1):
        for bv in (0, 1):
            ok = ((lo_a <= av) & (hi_a >= av) & (lo_b <= bv) & (hi_b >= bv)).float()
            v = T[:, 2 * av + bv]
            lo = torch.where(ok > 0, torch.minimum(lo, v.expand_as(lo)), lo)
            hi = torch.where(ok > 0, torch.maximum(hi, v.expand_as(hi)), hi)
    return lo, hi
