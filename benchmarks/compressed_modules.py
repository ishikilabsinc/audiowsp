import torch
import torch.nn as nn
import torch.nn.functional as F

class TuckerPosConv1d(nn.Module):
    '''16-group temporal convolution factorized via mode-3 Tucker SVD: (1024, 64, r_t) core + (16, 128, r_t) temporal basis.'''
    def __init__(self, core, U_time, bias):
        super().__init__()
        self.core   = nn.Parameter(core)   # (1024, 64, r_t)
        self.U_time = nn.Parameter(U_time) # (16, 128, r_t)
        self.bias   = nn.Parameter(bias)   # (1024,)

    def forward(self, x):
        # x: (B, 1024, T)
        W = torch.einsum('gik,gjk->gij', self.core.view(16, 64*64, -1), self.U_time).reshape(1024, 64, 128)
        out = F.conv1d(x, W.type_as(x), self.bias.type_as(x), padding=64, groups=16)[:, :, :-1]
        return F.gelu(out)

class CompressedTransformerLayer(nn.Module):
    def __init__(self, ln1_w, ln1_b, H_keep, qk_factors, v_factors, o_factors, ln2_w, ln2_b, ffn_mode, ffn_factors):
        super().__init__()
        self.H_keep = H_keep
        self.D_h = H_keep * 64
        self.ln1_w = nn.Parameter(ln1_w)
        self.ln1_b = nn.Parameter(ln1_b)

        if len(qk_factors) == 2:
            self.qk_is_lr = False
            self.qk_W = nn.Parameter(qk_factors[0])
            self.qk_b = nn.Parameter(qk_factors[1])
        else:
            self.qk_is_lr = True
            self.qk_U = nn.Parameter(qk_factors[0])
            self.qk_V = nn.Parameter(qk_factors[1])
            self.qk_b = nn.Parameter(qk_factors[2])

        if len(v_factors) == 2:
            self.v_is_lr = False
            self.v_W = nn.Parameter(v_factors[0])
            self.v_b = nn.Parameter(v_factors[1])
        else:
            self.v_is_lr = True
            self.v_U = nn.Parameter(v_factors[0])
            self.v_V = nn.Parameter(v_factors[1])
            self.v_b = nn.Parameter(v_factors[2])

        if len(o_factors) == 2:
            self.o_is_lr = False
            self.o_W = nn.Parameter(o_factors[0])
            self.o_b = nn.Parameter(o_factors[1])
        else:
            self.o_is_lr = True
            self.o_U = nn.Parameter(o_factors[0])
            self.o_V = nn.Parameter(o_factors[1])
            self.o_b = nn.Parameter(o_factors[2])

        self.ln2_w = nn.Parameter(ln2_w)
        self.ln2_b = nn.Parameter(ln2_b)
        self.ffn_mode = ffn_mode
        if ffn_mode == 'full':
            self.fc1_W = nn.Parameter(ffn_factors[0])
            self.fc1_b = nn.Parameter(ffn_factors[1])
            self.fc2_W = nn.Parameter(ffn_factors[2])
            self.fc2_b = nn.Parameter(ffn_factors[3])
        elif ffn_mode == 'lin':
            self.lin_U = nn.Parameter(ffn_factors[0])
            self.lin_V = nn.Parameter(ffn_factors[1])
            self.lin_b = nn.Parameter(ffn_factors[2])
        elif ffn_mode == 'topk_res':
            self.fc1_W = nn.Parameter(ffn_factors[0])
            self.fc1_b = nn.Parameter(ffn_factors[1])
            self.fc2_W = nn.Parameter(ffn_factors[2])
            self.res_U = nn.Parameter(ffn_factors[3])
            self.res_V = nn.Parameter(ffn_factors[4])
            self.comb_b = nn.Parameter(ffn_factors[5])

    def forward(self, x):
        B, T, _ = x.shape
        ln1 = F.layer_norm(x.float(), (1024,), self.ln1_w.float(), self.ln1_b.float()).type_as(x)
        if self.qk_is_lr:
            qk = F.linear(F.linear(ln1, self.qk_V.type_as(x)), self.qk_U.type_as(x), self.qk_b.type_as(x))
        else:
            qk = F.linear(ln1, self.qk_W.type_as(x), self.qk_b.type_as(x))
        q, k = qk[:, :, :self.D_h], qk[:, :, self.D_h:]

        if self.v_is_lr:
            v = F.linear(F.linear(ln1, self.v_V.type_as(x)), self.v_U.type_as(x), self.v_b.type_as(x))
        else:
            v = F.linear(ln1, self.v_W.type_as(x), self.v_b.type_as(x))

        q = q.view(B, T, self.H_keep, 64).transpose(1, 2)
        k = k.view(B, T, self.H_keep, 64).transpose(1, 2)
        v = v.view(B, T, self.H_keep, 64).transpose(1, 2)
        at = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, self.D_h)

        if self.o_is_lr:
            attn_out = F.linear(F.linear(at, self.o_V.type_as(x)), self.o_U.type_as(x), self.o_b.type_as(x))
        else:
            attn_out = F.linear(at, self.o_W.type_as(x), self.o_b.type_as(x))
        xmid = x + attn_out

        ln2 = F.layer_norm(xmid.float(), (1024,), self.ln2_w.float(), self.ln2_b.float()).type_as(xmid)
        if self.ffn_mode == 'full':
            ffn_out = F.linear(F.gelu(F.linear(ln2, self.fc1_W.type_as(xmid), self.fc1_b.type_as(xmid))), self.fc2_W.type_as(xmid), self.fc2_b.type_as(xmid))
        elif self.ffn_mode == 'lin':
            ffn_out = F.linear(F.linear(ln2, self.lin_V.type_as(xmid)), self.lin_U.type_as(xmid), self.lin_b.type_as(xmid))
        elif self.ffn_mode == 'topk_res':
            h = F.gelu(F.linear(ln2, self.fc1_W.type_as(xmid), self.fc1_b.type_as(xmid)))
            ffn_out = F.linear(h, self.fc2_W.type_as(xmid)) + F.linear(F.linear(ln2, self.res_V.type_as(xmid)), self.res_U.type_as(xmid), self.comb_b.type_as(xmid))
        return xmid + ffn_out

class Rank64Bridge(nn.Module):
    def __init__(self, U, V, b):
        super().__init__()
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.b = nn.Parameter(b)

    def forward(self, x):
        ln = F.layer_norm(x.float(), (1024,)).type_as(x)
        return x + F.linear(F.linear(ln, self.V.type_as(x)), self.U.type_as(x), self.b.type_as(x))
