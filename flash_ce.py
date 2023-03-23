import torch

def split_all(split, *tensors):
    tsplits = [t.split(split) for t in tensors]
    split_sizes = [t.shape[0] for t in tsplits[0]]
    
    acc = 0
    starts = []
    sizes = []
    for s in split_sizes:
        starts.append(acc)
        sizes.append(s)
        acc += s
    
    return list(zip(*[starts, sizes, *tsplits]))



def ce_p(left, right, mask, rix):
    """
    left : L x D
    right : R x D
    mask : L
    rix : L
    
    out: 2 x L
    """
    
    scores = left @ right.t()
    ret = scores.new_empty((2, left.shape[0]))
    
    ret[0] = scores.logsumexp(1)
    ret[1] = torch.where(
        mask,
        scores.gather(1, rix.unsqueeze(1)).squeeze(),
        0
    )
    
    return ret


def ce_fold(wns):
    """
    wpns : N x 2 x L
    out: 2 x L
    """
    ws, ns = wns.unbind(1)
    w = ws.logsumexp(0)
    n = ns.sum(0)
    return torch.stack((w, n))

def ce_q(wn):
    return wn[0] - wn[1]

class FlashCE(torch.autograd.Function):
    @staticmethod
    def forward(left, right, truth, lsplit=1024, rsplit=1024, fold_at=1024):
        """
        left  : L x D  (real)
        right : R x D  (real)
        truth : L      (index into R)
        
        returns: L-vector with per-sample CE-loss
        """

        lts = split_all(lsplit, left, truth)
        rs = split_all(rsplit, right)

        ret = []
        buffer = left.new_empty((fold_at, 2, lsplit)) #Buffer for folding.

        wn = left.new_empty(2, left.shape[0])
        
        lix = torch.arange(lsplit)
        
        for l_start, l_size, l, t in lts:
            bix = 0
            for r_start, r_size, r in rs:
                if bix == fold_at: # If we've filled up the folding-buffer: Fold
                    buffer[0] = ce_fold(buffer)
                    bix = 1
                tshift = t - r_start
                mask = (0 <= tshift) & (tshift < r_size)
                rix = torch.where(
                    mask,
                    tshift,
                    0
                )

                buffer[bix] = ce_p(l, r, mask, rix)
                bix += 1
            wn[:, l_start:l_start+l_size] = ce_fold(buffer[:bix])

        return ce_q(wn), wn[0]
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        left, right, truth, lsplit, rsplit, _ = inputs
        _, w = outputs

        ctx.save_for_backward(left, right, truth, w)
        ctx.splits = (lsplit, rsplit)
        
    @staticmethod
    def backward(ctx, grad_output, grad_w):
        left, right, truth, ws = ctx.saved_tensors
        
        lsplit, rsplit = ctx.splits
        
        ret_dl = left.new_zeros(left.shape)
        ret_dr = right.new_zeros(right.shape)
        
        ltdwg = split_all(lsplit, left, truth, ret_dl, ws, grad_output)
        rd = split_all(rsplit, right, ret_dr)
        
        for l_start, l_size, l, t, dl, w, g in ltdwg:
            for r_start, r_size, r, dr in rd:
                tshift = t - r_start
                mask = (0 <= tshift) & (tshift < r_size)
                lix = torch.arange(l_size)[mask]
                rix = tshift[mask]
                dq = (l @ r.t() - w.unsqueeze(1)).exp()
                dq[lix, rix] -= 1
                dq *= g.unsqueeze(1)
                dl += dq @ r
                dr += dq.t() @ l
        
        return ret_dl, ret_dr, None, None, None, None
    
def ce(left, right, truth, lsplit=1024, rsplit=1024, fold_at=2):
    ret, _ = FlashCE.apply(left, right, truth, lsplit, rsplit, fold_at)
    return ret



def check_grad(L=6, R=12, D=3, lsplit=2, rsplit=5, fold_at=2):
    inputs = (
            torch.randn(L, D, dtype=torch.double, requires_grad=True),
            torch.randn(R, D, dtype=torch.double, requires_grad=True),
            torch.randint(R, (L,)),
            lsplit, 
            rsplit,
            fold_at
            )
    return torch.autograd.gradcheck(lambda *x: ce(*x).mean(), inputs)

def check_valid(L=6, R=12, D=3, lsplit=2, rsplit=5, fold_at=2):


    l = torch.randn(L, D)
    r = torch.randn(R, D)
    t = torch.randint(R, (L,))

    A = ce(l, r, t, lsplit, rsplit, fold_at)
    B = torch.nn.functional.cross_entropy(l @ r.t(), t, reduction='none')
    return torch.allclose(A, B)
