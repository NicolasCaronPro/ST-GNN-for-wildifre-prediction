
import torch
import torch.nn as nn
from forecasting_models.pytorch.distillation_utils import RelationAttention

def test_relation_attention():
    print("Testing RelationAttention...")
    
    # Parameters
    num_teachers = 3
    num_classes = 5
    embedding_dim = 10
    hidden_dim = 16
    num_heads = 4
    batch_size = 32
    
    # Instantiate model
    model = RelationAttention(
        num_teachers=num_teachers,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    
    # Test case 1: [M, B, C]
    print("Test case 1: Input shape [M, B, C]")
    teacher_logits = torch.randn(num_teachers, batch_size, num_classes)
    output = model(teacher_logits)
    assert output.shape == (batch_size, embedding_dim), f"Expected {(batch_size, embedding_dim)}, got {output.shape}"
    print("Passed!")
    
    # Test case 2: [B, C] (Single teacher case)
    print("Test case 2: Input shape [B, C]")
    # Note: RelationAttention expects num_teachers to match. 
    # If we pass [B, C], it treats it as M=1.
    # So we need a model initialized with num_teachers=1 for this test to be fully valid logic-wise,
    # or we accept that it will fail the assertion M == self.num_teachers if initialized with 3.
    # Let's re-instantiate for single teacher
    model_single = RelationAttention(
        num_teachers=1,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    teacher_logits_single = torch.randn(batch_size, num_classes)
    output_single = model_single(teacher_logits_single)
    assert output_single.shape == (batch_size, embedding_dim), f"Expected {(batch_size, embedding_dim)}, got {output_single.shape}"
    print("Passed!")

    # Test case 3: [M, B, 1, C]
    print("Test case 3: Input shape [M, B, 1, C]")
    teacher_logits_4d = torch.randn(num_teachers, batch_size, 1, num_classes)
    output_4d = model(teacher_logits_4d)
    assert output_4d.shape == (batch_size, embedding_dim), f"Expected {(batch_size, embedding_dim)}, got {output_4d.shape}"
    print("Passed!")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_relation_attention()
