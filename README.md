# L-Reg_NeurIPS24
Official Code of **Interpret Your Decision: Logical Reasoning Regularization for Generalization in Visual Classification (NeurIPS 2024 Spotlight)**

As many friends requested, I post the L-Reg here:

'''
loss = other_mainly_used_loss_for_this_task
weight = weight_of_L_Reg
# feat is the logit output of a layer from the model. 
# as we show in the paper, if the dimensions of logit are independent of each other, it will be better. 
A = torch.matmul(logit.permute(1, 0), feat)
# M, b  @  b, N -> M * N
A = F.softmax(A, dim=1) # M, N, this is important
A_entropy = (
        A.mean(0)
        * torch.log(A.mean(0) + 1e-12)
    ).sum()
A_dim_entropy = (
            -((A + 1e-12)
                    * torch.log(
                A + 1e-12
            )
            )
            .sum(1) # 《
            .mean()
    )
L_Reg = A_entropy + A_dim_entropy 
loss += L_Reg * weight
# Done! Feel free to have a try :P
'''

