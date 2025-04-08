import pandas as pd

def get_decode_phase_shapes(seq_len_cache = 126,  # Number of cached tokens (previously decoded tokens)
                            batch_size = 1,
                            h_dim = 5120,
                            num_heads = 40,
                            intermediate_dim = 13824,
                            file_path = '../gamma/data/model/decode_phase_shapes.csv'):
    """
    Given the sequence length of cached tokens (seq_len_cache), batch size, hidden dimension, and number of heads,
    output MxK and KxN for all matrix multiplications in the decode phase.
    
    Parameters:
    - seq_len_cache: Length of cached tokens (previously decoded tokens in the cache)
    - batch_size: Size of the batch (usually 1 during decoding, but can handle larger batch sizes)
    - h_dim: Hidden dimension of the model
    - num_heads: Number of attention heads
    
    Returns:
    - List of tuples (M, K, N) for each matrix multiplication in the decode phase.
    """
    head_dim = h_dim // num_heads  # Dimension of each attention head

    shapes = []
    shapes1 = []

    # Matrix multiplication dimensions for Attention Projections (Q, K, V)
    # Input shape: (batch_size, 1, h_dim) since one token at a time
    # Query projection: (B, 1, H) x (H, H) -> (B, 1, H)
    shapes.append(("Query Projection", batch_size, h_dim, h_dim))
    shapes1.append([batch_size, h_dim, h_dim, '1', '1', '1', '3'])

    # Key/Value projections: (B, 1, H) x (H, H) -> (B, 1, H)
    shapes.append(("Key Projection", batch_size, h_dim, h_dim))
    shapes.append(("Value Projection", batch_size, h_dim, h_dim))

    # Attention score calculation: Query (B, 1, H) x Key Cache (B, seq_len_cache, H) -> (B, 1, seq_len_cache)
    shapes.append(("Attention Scores", batch_size, head_dim, seq_len_cache))
    shapes1.append([batch_size, seq_len_cache, head_dim, '1', '1', '1', '3'])

    # Attention output: Attention weights (B, 1, seq_len_cache) x Value Cache (B, seq_len_cache, H) -> (B, 1, H)
    shapes.append(("Attention Output", batch_size, seq_len_cache, head_dim))
    shapes1.append([batch_size, head_dim, seq_len_cache, '1', '1', '1', '3'])

    # Attention output projection: (B, 1, H) x (H, H) -> (B, 1, H)
    shapes.append(("Attention Output Projection", batch_size, h_dim, h_dim))

    # Feed-Forward Up projection: (B, 1, H) -> (B, 1, 4H)
    # (B, 1, H) x (H, 4H) -> (B, 1, 4H)
    shapes.append(("FFN Up Projection", batch_size, h_dim, intermediate_dim))
    shapes1.append([batch_size, intermediate_dim, h_dim, '1', '1', '1', '3'])

    # Feed-Forward Down projection: (B, 1, 4H) -> (B, 1, H)
    # (B, 1, 4H) x (4H, H) -> (B, 1, H)
    shapes.append(("FFN Down Projection", batch_size, intermediate_dim, h_dim))
    shapes1.append([batch_size, h_dim, intermediate_dim, '1', '1', '1', '3'])

    ### save shapes as a csv file
    column_names = ['M', 'N', 'K', '0', '0', '0', 'T']
    shapes_df = pd.DataFrame(shapes1, columns=column_names)
    shapes_df.to_csv(file_path, index=False)


def get_fine_tune_shapes(seq_len=4096,
                          batch_size=1,
                          h_dim=5120,
                          num_heads=40,
                          intermediate_dim=13824,
                          file_path='../gamma/data/model/finetuning_shapes.csv'):
    """
    Generate matrix shapes for forward and backward passes during LLM fine-tuning,
    assuming a standard feed-forward network without KV cache.
    
    Parameters:
    - seq_len: Sequence length
    - batch_size: Size of the batch
    - h_dim: Hidden dimension of the model
    - num_heads: Number of attention heads
    - intermediate_dim: Dimension of the intermediate layer in FFN
    - file_path: Path to save the CSV file
    
    Returns:
    - List of dictionaries containing shape information for each operation
    """
    head_dim = h_dim // num_heads

    # shapes = []

    # def add_shape(name, m, n, k, forward=True, backward=True, backward_weight=True):
    #     shapes.append({
    #         'Name': name,
    #         'M': m,
    #         'N': n,
    #         'K': k,
    #         'Forward': int(forward),
    #         'Backward_grad': int(backward),
    #         'Backward_weight': int(backward_weight),
    #         'Type': 3  # Assuming all operations are matrix multiplications
    #     })

    # Forward pass
    shape = []
    # add_shape("Query Projection", batch_size * seq_len, h_dim, h_dim)
    shape.append([batch_size * seq_len, h_dim, h_dim, '1', '1', '1', '3']) ### QKVO
    shape.append([batch_size * seq_len, seq_len, head_dim, '1', '1', '1', '3']) ### QKT
    shape.append([batch_size * seq_len, head_dim, seq_len, '1', '1', '1', '3']) ### QKTV
    shape.append([batch_size * seq_len, intermediate_dim, h_dim, '1', '1', '1', '3']) ### FFN up
    shape.append([batch_size * seq_len, h_dim, intermediate_dim, '1', '1', '1', '3']) ### FFN down

    # Backward pass
    # Gradient with respect to inputs
    shape.append([batch_size * seq_len, intermediate_dim, h_dim, '1', '1', '1', 3]) ### FFN down backprop grad ## weight store
    shape.append([batch_size * h_dim, intermediate_dim, seq_len, '1', '1', '1', 3]) ### FFN up/gate/down backprop current ## input store
    shape.append([batch_size * seq_len, h_dim, intermediate_dim, '1', '1', '1', 3]) ### FFN up backprop grad  ## weight store
    shape.append([batch_size * seq_len, h_dim, seq_len, '1', '1', '1', 3]) ### Attention output backprop grad (df/dq, df/dk)  ## 
    shape.append([batch_size * seq_len, head_dim, seq_len, '1', '1', '1', 3]) ### Attention output backprop grad (df/dv)
    shape.append([batch_size * h_dim, h_dim, seq_len, '1', '1', '1', 3]) ### Wq, Wk, Wv ### input store
    shape.append([batch_size * seq_len, h_dim, h_dim, '1', '1', '1', 3]) ### hidden embedding
    # add_shape("Key Projection Grad", batch_size * seq_len, h_dim, h_dim, forward=False, backward_weight=False)
    # add_shape("Query Projection Grad", batch_size * seq_len, h_dim, h_dim, forward=False, backward_weight=False)

    # Gradient with respect to weights
    # add_shape("FFN Down Projection Weight Grad", h_dim, intermediate_dim, batch_size * seq_len, forward=False, backward=False)
    # add_shape("FFN Up Projection Weight Grad", intermediate_dim, h_dim, batch_size * seq_len, forward=False, backward=False)
    # add_shape("Attention Output Projection Weight Grad", h_dim, h_dim, batch_size * seq_len, forward=False, backward=False)
    # add_shape("Value Projection Weight Grad", h_dim, h_dim, batch_size * seq_len, forward=False, backward=False)
    # add_shape("Key Projection Weight Grad", h_dim, h_dim, batch_size * seq_len, forward=False, backward=False)
    # add_shape("Query Projection Weight Grad", h_dim, h_dim, batch_size * seq_len, forward=False, backward=False)

    # Save shapes as a CSV file
    column_names = ['M', 'N', 'K', '0', '0', '0', 'T']
    shapes_df = pd.DataFrame(shape, columns=column_names)
    shapes_df.to_csv(file_path, index=False)

    return shape

    
def get_prefill_shapes(seq_len=4096,
                          batch_size=1,
                          h_dim=5120,
                          num_heads=40,
                          intermediate_dim=13824,
                          file_path='../gamma/data/model/prefill_shapes.csv'):
    """
    Generate matrix shapes for forward and backward passes during LLM prefilling,
    assuming a standard feed-forward network without KV cache.
    
    Parameters:
    - seq_len: Sequence length
    - batch_size: Size of the batch
    - h_dim: Hidden dimension of the model
    - num_heads: Number of attention heads
    - intermediate_dim: Dimension of the intermediate layer in FFN
    - file_path: Path to save the CSV file
    
    Returns:
    - List of dictionaries containing shape information for each operation
    """
    head_dim = h_dim // num_heads

    # shapes = []

    # def add_shape(name, m, n, k, forward=True, backward=True, backward_weight=True):
    #     shapes.append({
    #         'Name': name,
    #         'M': m,
    #         'N': n,
    #         'K': k,
    #         'Forward': int(forward),
    #         'Backward_grad': int(backward),
    #         'Backward_weight': int(backward_weight),
    #         'Type': 3  # Assuming all operations are matrix multiplications
    #     })

    # Forward pass
    shape = []
    # add_shape("Query Projection", batch_size * seq_len, h_dim, h_dim)
    shape.append([batch_size * seq_len, h_dim, h_dim, '1', '1', '1', '3']) ### QKVO
    shape.append([batch_size * seq_len, seq_len, head_dim, '1', '1', '1', '3']) ### QKT
    shape.append([batch_size * seq_len, head_dim, seq_len, '1', '1', '1', '3']) ### QKTV
    shape.append([batch_size * seq_len, intermediate_dim, h_dim, '1', '1', '1', '3']) ### FFN up
    shape.append([batch_size * seq_len, h_dim, intermediate_dim, '1', '1', '1', '3']) ### FFN down

    column_names = ['M', 'N', 'K', '0', '0', '0', 'T']
    shapes_df = pd.DataFrame(shape, columns=column_names)
    shapes_df.to_csv(file_path, index=False)

    return shape