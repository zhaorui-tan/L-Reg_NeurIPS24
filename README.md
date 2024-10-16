# L-Reg_NeurIPS24
Official Code of **Interpret Your Decision: Logical Reasoning Regularization for Generalization in Visual Classification (NeurIPS 2024 Spotlight)**
The preprint version of the paper is available at: https://arxiv.org/abs/2410.04492




As many friends requested, I post the L-Reg here:

```python
loss = other_mainly_used_loss_for_this_task
weight = weight_of_L_Reg

# feat is the logit output of a layer from the model. 
# as we show in the paper, if the dimensions of logit are independent of each other, it will be better. 
A = torch.matmul(logit.permute(1, 0), feat)

# M, b  @  b, N -> M * N
A = F.softmax(A, dim=1) # M, N, this is important

# now calculating L_Reg
A_p1 = (
        A.mean(0) * torch.log(A.mean(0) + 1e-12)
    ).sum()
A_p2 = (
        -((A + 1e-12) * torch.log( A + 1e-12))
        .sum(1)
        .mean()
    )
L_Reg = A_p1 + A_p2

# add it back to other losses
loss += L_Reg * weight
# Done! Feel free to have a try :P
```

Someone may be interested in the solution of limitations of L-Reg, I offer the Ortho-Reg code here as well:

```python
weight = weight_of_ortho_reg

# the calculation begins here
feat_ = feat / feat.norm(dim=-1, keepdim=True) # normalize at the first
B = feat_.T @ feat_
id_matrix = torch.eye(*B.size(), out=torch.empty_like(B)).cuda() * 1.
ortho_reg = nn.MSELoss()(id_matrix, B)

# add it back to other losses, especially for L-Reg
loss += ortho_reg * weight
# Done! 
```
