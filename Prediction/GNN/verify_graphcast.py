
import torch
import dgl
import sys
import os

# Add parent directory to path to find forecasting_models
sys.path.append('/Home/Users/ncaron/WORK/GNN')

from forecasting_models.pytorch.models import GraphCast

def test_graphcast_horizon():
    print("Testing GraphCast with horizon > 0...")
    
    # Parameters
    in_channels = 10
    horizon = 3
    n_sequences = 5
    end_channels = 8
    out_channels = 2
    input_dim_grid_nodes = 16
    
    # Instantiate model
    model = GraphCast(
        input_dim_grid_nodes=input_dim_grid_nodes,
        end_channels=end_channels,
        out_channels=out_channels,
        n_sequences=n_sequences,
        horizon=horizon,
        task_type='regression'
    )
    
    print("Model instantiated successfully.")
    
    # Check if input_dim_grid_nodes was increased in GraphCastNet
    # GraphCastNet stores input_dim_grid_nodes? 
    # Let's check model.net attributes if possible, or just rely on forward pass not crashing.
    
    # Create dummy inputs
    # GraphCast forward: X, graph, graph2mesh, mesh2graph, z_prev=None
    # X shape: (batch, seq_len, in_channels, n_nodes) -> Permuted to (in_channels, batch, seq_len) ?
    # Wait, let's check forward implementation again.
    # X = X.permute(2, 0, 1)
    # Input X to forward seems to be (Batch, Nodes, Channels) or something?
    # Original code:
    # X: Tensor shaped (batch, seq_len, in_channels, n_nodes)
    # But line 1940: X = X.permute(2, 0, 1)
    # If input is 4D, permute(2,0,1) would fail or do something weird if not 3D.
    # Let's look at Training.launch_batch again.
    # inputs_horizon = self.compute_inputs(...)
    # If inputs_horizon is 3D: (Batch, Nodes, Time) ?
    # compute_inputs returns inputs_horizon.
    # If inputs is 3D (Batch, Nodes, Time), inputs_horizon is (Batch, Nodes, Time_slice).
    
    # GraphCast forward:
    # X = X.permute(2, 0, 1) -> (Time, Batch, Nodes) ?
    # If X is (Batch, Nodes, Time)
    # permute(2, 0, 1) -> (Time, Batch, Nodes)
    
    # Then z_prev is (Batch, EndChannels, Time) ?
    # line 1943: z_prev = torch.zeros((X.shape[1], self.end_channels, self.n_sequences), ...)
    # X.shape[1] is Batch (after permute).
    
    # So X input to forward should be (Batch, Nodes, Time).
    
    batch_size = 2
    n_nodes = 4
    
    X = torch.randn(batch_size, n_nodes, in_channels) # Wait, in_channels is usually time?
    # Let's assume X is (Batch, Nodes, Features)
    
    # Let's check Training.launch_batch again.
    # inputs_horizon = inputs[:, :, H - self.ks:H + 1] -> (Batch, Nodes, Time)
    
    # So X is (Batch, Nodes, Time).
    # But wait, GraphCast expects (Time, Batch, Nodes) ?
    # If X is (Batch, Nodes, Time)
    # permute(2, 0, 1) -> (Time, Batch, Nodes)
    
    # Then z_prev is (Batch, EndChannels, Time) ?
    # In Training.launch_batch:
    # z_prev = torch.stack(history, dim=2) -> (Batch, EndChannels, Time)
    
    # In GraphCast.forward (my fix):
    # X = X.permute(2, 0, 1) -> (Time, Batch, Nodes)
    # z_prev is (Batch, EndChannels, Time)
    # torch.cat((X, z_prev), dim=1) -> This will fail if dimensions don't match.
    # X is (Time, Batch, Nodes)
    # z_prev is (Batch, EndChannels, Time)
    
    # Wait, I might have messed up the dimension logic in my fix or understanding.
    # Let's re-read GraphCast.forward carefully.
    
    # Original:
    # X = X.permute(2, 0, 1)
    # ...
    # x = self.net(X, ...)[-1]
    
    # If X is (Batch, Nodes, Time)
    # permute(2, 0, 1) -> (Time, Batch, Nodes)
    
    # z_prev is (Batch, EndChannels, Time)
    
    # If I want to concat z_prev to X, I need to align them.
    # If X represents features on nodes, and z_prev represents features on nodes (or global?),
    # z_prev from Training is (Batch, Hidden, Time).
    # It seems z_prev is per-graph (or per-node if Batch is nodes?).
    
    # If Batch is actually Batch*Nodes (node-level), then z_prev is (Batch*Nodes, Hidden, Time).
    # Then X is (Batch*Nodes, 1, Time) ? No.
    
    # Let's assume standard case:
    # X: (Batch, Nodes, Features/Time)
    
    # If I look at GRU:
    # X = torch.cat((X, z_prev), dim=1)
    # X is (Batch, Channels, Time) ?
    # In GRU forward:
    # z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
    # X = torch.cat((X, z_prev), dim=1)
    
    # So X and z_prev have same shape except dim 1.
    # (Batch, Channels, Time)
    
    # Now GraphCast.
    # X input is likely (Batch, Nodes, Time) or (Batch, Features, Time) ?
    # If it is (Batch, Nodes, Time), then dim 1 is Nodes.
    # z_prev is (Batch, EndChannels, Time).
    # Concatenating on dim 1 would mean concatenating Nodes and EndChannels? That seems wrong.
    
    # Unless X is (Batch, Features, Time).
    
    # Let's check `compute_inputs` in `pytorch_model.py`.
    # inputs_horizon = inputs[:, :, H - self.ks:H + 1]
    # If inputs is (Batch, Features, Time), then inputs_horizon is (Batch, Features, Time).
    
    # So X is (Batch, Features, Time).
    
    # GraphCast.forward:
    # X = X.permute(2, 0, 1) -> (Time, Batch, Features)
    
    # My fix:
    # if self.horizon > 0:
    #    X = torch.cat((X, z_prev), dim=1)
    
    # If X is (Batch, Features, Time) (before permute)
    # and z_prev is (Batch, EndChannels, Time)
    # Then cat(dim=1) results in (Batch, Features+EndChannels, Time).
    
    # Then X = X.permute(2, 0, 1) -> (Time, Batch, Features+EndChannels).
    
    # This matches `input_dim_grid_nodes` increase if `input_dim_grid_nodes` corresponds to Features.
    # GraphCastNet(input_dim_grid_nodes=...)
    
    # So the assumption is X is (Batch, Features, Time).
    
    # Let's verify this assumption by running the script.
    
    B = 2
    F = input_dim_grid_nodes
    T = n_sequences
    
    X = torch.randn(B, F, T)
    z_prev = torch.randn(B, end_channels, T)
    
    # Mock GraphCastNet
    class MockNet(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            
        def forward(self, x, g, g2m, m2g):
            # x should be (Time, Batch, Features)
            # Check last dimension
            if x.shape[2] != self.input_dim:
                raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[2]}")
            return [x] # Return list
            
    # We need to access the instantiated GraphCastNet to check its input_dim_grid_nodes
    # But we can't easily.
    # However, we can replace model.net with our MockNet.
    
    expected_dim = input_dim_grid_nodes + end_channels
    model.net = MockNet(expected_dim)
    
    try:
        output, logits, hidden = model(X, None, None, None, z_prev=z_prev)
        print("Forward pass successful.")
        
        # Verify z_prev usage
        # Since we cat z_prev, if we change z_prev, input to net changes.
        # And since net returns input, output changes.
        
        output1, _, _ = model(X, None, None, None, z_prev=z_prev)
        output2, _, _ = model(X, None, None, None, z_prev=torch.zeros_like(z_prev))
        
        # output is (Time, Batch, Features+EndChannels) passed through linear layers.
        # So it should be different.
        
        if not torch.allclose(output1, output2):
             print("SUCCESS: z_prev influences output.")
        else:
             print("FAILURE: z_prev does not influence output.")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graphcast_horizon()
