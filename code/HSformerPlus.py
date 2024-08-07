import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
       
        super().__init__()
        
        #...

        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0: # Vertical strips
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1: # Horizontal strips
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.H_sp_ = self.H_sp
        self.W_sp_ = self.W_sp

        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def compute_rpe(self, polar_coord, B, H, W):
        if self.idx == 1:

            theta_pe = self.Hrelative_dis_compute(polar_coord[..., 0], polar_coord[..., 0], polar_coord[..., 1])[0, ...]

            rpe = theta_pe.view(W // self.W_sp, H // self.H_sp, self.H_sp * self.W_sp, self.H_sp * self.W_sp).unsqueeze(0).unsqueeze(0)
            rpe_h = rpe.repeat(B, self.num_heads, 1, 1, 1, 1).clone()  # (b, head, w // w_sp, h * w_sp, h * w_sp)

            rpe_h = rpe_h.reshape(-1, self.num_heads, self.H_sp * self.W_sp, self.H_sp * self.W_sp).contiguous()
            rpe = rpe_h.clone().cuda()

        elif self.idx == 0:

            phi_pe = self.Vrelative_dis_compute(polar_coord[..., 1], polar_coord[..., 1])[:, 0, ...]

            rpe = phi_pe.view(H // self.H_sp, W // self.W_sp, self.H_sp * self.W_sp, self.H_sp * self.W_sp).unsqueeze(0).unsqueeze(0)
            rpe_v = rpe.repeat(B, self.num_heads, 1, 1, 1, 1).clone()  # (b, head, w // w_sp, h * w_sp, h * w_sp)

            rpe_v = rpe_v.reshape(-1, self.num_heads, self.H_sp * self.W_sp, self.H_sp * self.W_sp).contiguous()
            rpe = rpe_v.clone().cuda()

        return rpe

    def forward(self, temp, polar_coord=None, x=None):
        """
        x: B N C
     mask: B N N
        """
        B, _, C, H, W = temp.shape

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            print("ERROR MODE in forward", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad)) 
        q, k, v = qkv[0], qkv[1], qkv[2]

        v, rpe = self.get_polar_rpe(v, idx, polar_coord)

        q = q * self.scale
        if idx == -1:
            attn = (q @ k.transpose(-2, -1))  
        else:
            score = (q @ k.transpose(-2, -1) + rpe) 
            attn_norm = torch.norm(score, p=1, dim=-1, keepdim=True).to(score.device)
            attn_l1 = score / (attn_norm + 1e-8)
            if idx == 1:
                attn = (1 - torch.cos(attn_l1 * (math.pi/2))) * (math.sin(math.pi/2) * math.sin(math.pi/2))
            elif idx == 0:
                attn = 1 - torch.cos(attn_l1 * (math.pi/2))

        x = (attn @ v) + rpe

        return x