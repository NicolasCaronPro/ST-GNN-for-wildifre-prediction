
import torch
import dgl
import sys
import os

# Add parent directory to path to find forecasting_models
sys.path.append('/Home/Users/ncaron/WORK/GNN')

from forecasting_models.pytorch.models import GraphCastGRU

def test_graphcast_gru_horizon():
    print("Testing GraphCastGRU with horizon > 0...")
    
    # Parameters
    in_channels = 10
    horizon = 3
    n_sequences = 5
    end_channels = 8
    out_channels = 2
    input_dim_grid_nodes = 16
    
    # Instantiate model
    model = GraphCastGRU(
        in_channels=in_channels,
        input_dim_grid_nodes=input_dim_grid_nodes,
        end_channels=end_channels,
        out_channels=out_channels,
        n_sequences=n_sequences,
        horizon=horizon,
        task_type='regression'
    )
    
    print("Model instantiated successfully.")
    
    # Check for removed attributes
    if hasattr(model, 'decoder'):
        print("FAILURE: 'decoder' attribute still exists!")
    else:
        print("SUCCESS: 'decoder' attribute correctly removed.")
        
    if hasattr(model, 'forward_horizon'):
        print("FAILURE: 'forward_horizon' method still exists!")
    else:
        print("SUCCESS: 'forward_horizon' method correctly removed.")

    # Create dummy inputs
    batch_size = 2
    n_nodes = 4
    # X shape: (batch, in_channels, seq_len, n_nodes) -> Wait, let's check forward signature
    # forward(self, X, graph, graph2mesh, mesh2graph, z_prev=None)
    # X: Tensor shaped (batch, seq_len, in_channels, n_nodes) ? 
    # In code: B, C_in, T = X.shape (after some permute? No, let's check code again)
    
    # Code says:
    # B, C_in, T = X.shape
    # So X is (Batch*Nodes, Channels, Time) ?
    # But docstring says: X: Tensor shaped (batch, seq_len, in_channels, n_nodes).
    # Let's check the forward implementation I read earlier.
    # line 2066: B, C_in, T = X.shape
    # So input X must be 3D.
    
    # Let's assume X is (Batch*Nodes, Channels, Time) based on B, C_in, T = X.shape
    
    B_total = batch_size * n_nodes
    X = torch.randn(B_total, in_channels, n_sequences)
    
    # z_prev
    z_prev = torch.randn(B_total, end_channels, n_sequences)
    
    # Dummy graphs (needed for GraphCastNet)
    # GraphCastNet expects dgl graphs.
    # We can create simple dummy graphs.
    
    # grid graph
    u, v = torch.tensor([0, 1]), torch.tensor([1, 2])
    graph = dgl.graph((u, v), num_nodes=n_nodes)
    # Add dummy features if needed by GraphCastNet? 
    # GraphCastNet usually expects features on nodes/edges.
    # But let's see if we can just pass minimal graphs.
    
    # For this test, we might mock GraphCastNet to avoid complex graph setup if possible.
    # But GraphCastNet is inside the model.
    
    # Let's try to run it. If GraphCastNet fails due to missing graph features, we'll know.
    # But we primarily want to check the GRU part and z_prev concatenation which happens BEFORE GraphCastNet.
    
    # Actually, we can just check if the code runs up to GraphCastNet.
    # Or we can mock self.net.
    
    model.net = torch.nn.Identity() # Mock GraphCastNet to return input
    # But GraphCastNet input is X_graphcast = gru_last[None, : ,:] -> (1, B*N, hidden)
    # Identity will return (1, B*N, hidden).
    # Then: x = self.net(...)[-1] -> this will fail if identity returns tensor.
    # GraphCastNet returns a list/tuple? 
    # x = self.net(...)[-1]
    
    class MockNet(torch.nn.Module):
        def forward(self, x, g, g2m, m2g):
            # Return a list where last element is x
            return [x]
            
    model.net = MockNet()
    
    # Run forward
    try:
        output, logits, hidden = model(X, None, None, None, z_prev=z_prev)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
        
        # Verify z_prev usage
        # We can't easily verify z_prev usage with MockNet unless we check input to MockNet.
        # The input to MockNet is output of GRU.
        # GRU input is X concatenated with z_prev.
        # So if we change z_prev, output should change.
        
        output1, _, _ = model(X, None, None, None, z_prev=z_prev)
        output2, _, _ = model(X, None, None, None, z_prev=torch.zeros_like(z_prev))
        
        if not torch.allclose(output1, output2):
             print("SUCCESS: z_prev influences output.")
        else:
             print("FAILURE: z_prev does not influence output (or GRU ignores it).")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graphcast_gru_horizon()
