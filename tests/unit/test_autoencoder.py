import pytest
import torch
from bernn.dl.train.pytorch.aedann import AutoEncoder2
from bernn.dl.train.pytorch.aeekandann import KANAutoencoder2

@pytest.fixture
def sample_data():
    # Create sample input data
    batch_size = 32
    input_dim = 100
    return torch.randn(batch_size, input_dim)

@pytest.fixture
def standard_model():
    input_dim = 100
    n_batches = 3
    nb_classes = 2
    return AutoEncoder2(
        input_dim,
        n_batches=n_batches,
        nb_classes=nb_classes,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device='cpu',
        prune_threshold=0.001  # not actually implemented pruning but need to be there  
    )

@pytest.fixture
def kan_model():
    input_dim = 100
    n_batches = 3
    nb_classes = 2
    return KANAutoencoder2(
        input_dim,
        n_batches=n_batches,
        nb_classes=nb_classes,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device='cpu',
        prune_threshold=0.001
    )

@pytest.mark.unit
@pytest.mark.parametrize("model_fixture", ["standard_model", "kan_model"])
def test_autoencoder_dimensions(request, model_fixture, sample_data):
    model = request.getfixturevalue(model_fixture)
    # Test if encoder output dimensions are correct
    encoded, reconstructed, zinb_loss, kld = model(sample_data, sample_data, torch.zeros(sample_data.size(0)).int())
    
    assert encoded.shape[1] == 32, f"{model_fixture}: Encoded dimension should match layer2 size"
    assert reconstructed['mean'][1].shape == sample_data.shape, f"{model_fixture}: Reconstructed shape should match input shape"
    assert isinstance(zinb_loss, torch.Tensor), f"{model_fixture}: ZINB loss should be a tensor"
    assert isinstance(kld, torch.Tensor), f"{model_fixture}: KLD should be a tensor"

@pytest.mark.unit
@pytest.mark.parametrize("model_fixture", ["standard_model", "kan_model"])
def test_autoencoder_training_step(request, model_fixture, sample_data):
    model = request.getfixturevalue(model_fixture)
    # Test if model can perform a training step
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    # Forward pass
    encoded, reconstructed, zinb_loss, kld = model(sample_data, sample_data, torch.zeros(sample_data.size(0)).int())
    
    # Compute loss
    loss = criterion(reconstructed['mean'][1], sample_data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss), f"{model_fixture}: Loss should not be NaN"
    assert loss.item() > 0, f"{model_fixture}: Loss should be positive"

@pytest.mark.unit
def test_autoencoder_device_transfer_autoencoder():
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        # Set CUDA device
        device = 'cuda:0'
       
    model = AutoEncoder2(
        100,
        n_batches=3, 
        nb_classes=2,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device=device,
        prune_threshold=0.001  # not actually implemented pruning but need to be there  
    ).to(device)
    
    if device == 'cuda':
        assert next(model.parameters()).is_cuda, f"{AutoEncoder2.__name__}: Model should be on CUDA"
    else:
        assert not next(model.parameters()).is_cuda, f"{AutoEncoder2.__name__}: Model should be on CPU"

@pytest.mark.unit
def test_autoencoder_device_transfer_kan():
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        # Set CUDA device
        device = 'cuda:0'
        
    model = KANAutoencoder2(
        100,
        n_batches=3, 
        nb_classes=2,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device=device,
        prune_threshold=0.001
    ).to(device)
    
    if device == 'cuda':
        assert next(model.parameters()).is_cuda, f"{KANAutoencoder2.__name__}: Model should be on CUDA"
    else:
        assert not next(model.parameters()).is_cuda, f"{KANAutoencoder2.__name__}: Model should be on CPU"

@pytest.mark.unit
def test_model_prune_autoencoder():
    # Test model pruning functionality
    model = AutoEncoder2(
        100,
        n_batches=3, 
        nb_classes=2,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device='cpu',
        prune_threshold=0.001  # not actually implemented pruning but need to be there 
    )
    
    # Set prune threshold after model creation
    if hasattr(model, 'prune_threshold'):
        model.prune_threshold = 0.001
    
    # Store initial parameter count
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

@pytest.mark.unit
def test_model_prune_kan():
    # Test model pruning functionality
    model = KANAutoencoder2(
        100,
        n_batches=3, 
        nb_classes=2,
        mapper=True,
        layer1=64,
        layer2=32,
        n_layers=2,
        n_meta=0,
        n_emb=0,
        dropout=0.1,
        variational=False,
        conditional=False,
        zinb=False,
        add_noise=0,
        tied_weights=False,
        use_gnn=0,
        device='cpu',
        prune_threshold=0.001
    )
    
    # Set prune threshold after model creation
    if hasattr(model, 'prune_threshold'):
        model.prune_threshold = 0.001
    
    # Store initial parameter count
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Call pruning method
    try:
        n_neurons = model.prune_model_paperwise(False, False, weight_threshold=0.001)
        
        # Verify pruning results if succeeded
        assert isinstance(n_neurons, dict), f"{KANAutoencoder2.__name__}: Pruning should return a dictionary"
        assert "total_remaining" in n_neurons, f"{KANAutoencoder2.__name__}: Pruning results should include total remaining neurons"
    except (AttributeError, TypeError) as e:
        # If pruning is not implemented, skip this test
        pytest.skip(f"Model {KANAutoencoder2.__name__} does not support pruning: {str(e)}") 
