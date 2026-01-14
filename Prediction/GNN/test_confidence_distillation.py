
import torch
import torch.nn as nn
import torch.nn.functional as F
from forecasting_models.pytorch.distillation_utils import confidence_distillation_loss

def test_confidence_loss():
    B, C = 4, 3
    D_s = 10
    D_t = 8
    K = 2
    
    student_logits = torch.randn(B, C)
    student_feat = torch.randn(B, D_s)
    
    teacher_logits_list = [torch.randn(B, C) for _ in range(K)]
    teacher_feat_list = [torch.randn(B, D_t) for _ in range(K)]
    
    labels = torch.randint(0, C, (B,))
    
    # Mock FitNets
    fitnets = nn.ModuleList([nn.Linear(D_s, D_t) for _ in range(K)])
    
    # Mock Teacher Classifiers
    teacher_classifiers = [nn.Linear(D_t, C) for _ in range(K)]
    
    criterion = nn.CrossEntropyLoss()
    
    loss = confidence_distillation_loss(
        student_logits=student_logits,
        student_feat=student_feat,
        teacher_logits_list=teacher_logits_list,
        teacher_feat_list=teacher_feat_list,
        labels=labels,
        fitnets=fitnets,
        teacher_classifiers=teacher_classifiers,
        alpha=1.0,
        beta=1.0,
        T=2.0
    )
    
    print(f"Loss: {loss.item()}")
    assert not torch.isnan(loss)
    assert loss.item() > 0

if __name__ == "__main__":
    test_confidence_loss()
