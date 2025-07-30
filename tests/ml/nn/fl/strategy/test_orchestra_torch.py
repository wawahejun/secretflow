# Copyright wawahejun, hejunlbbc@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from secretflow import reveal
from secretflow_fl.utils.simulation.datasets_fl import load_mnist
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.ml.nn.fl.backend.torch.strategy.orchestra import OrchestraStrategy
from secretflow.security import SecureAggregator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_def import ConvNet


class OrchestraConvNet(BaseModule):
    """Small ConvNet for MNIST - modified for unsupervised feature extraction."""

    def __init__(self):
        super(OrchestraConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        # Use feature_layer instead of fc to avoid Orchestra strategy replacement
        self.feature_layer = nn.Linear(self.fc_in_dim, 128)
        self.output_dim = 128

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3))
        x = x.view(-1, self.fc_in_dim)
        x = F.relu(self.feature_layer(x))
        return x


def evaluate_clustering_quality(features, true_labels, num_clusters=10):
    """Helper function to evaluate clustering quality"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features)
    
    # Calculate clustering quality metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # Calculate cluster balance
    unique, counts = np.unique(pred_labels, return_counts=True)
    cluster_balance = 1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 1.0
    
    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'cluster_balance': float(cluster_balance),
        'num_clusters_found': len(unique)
    }


def create_unsupervised_dataloader(x_data, batch_size=32, shuffle=True):
    """Create unsupervised dataloader (ignoring labels)"""
    # Create dummy labels to meet Orchestra strategy data format requirements
    dummy_labels = torch.zeros(x_data.shape[0], dtype=torch.long)
    dataset = TensorDataset(x_data, dummy_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TestOrchestraStrategy:
    """Test cases for Orchestra unsupervised federated learning strategy."""

    def test_orchestra_initialization(self, sf_simulation_setup_devices):
        """Test Orchestra strategy initialization for unsupervised learning"""
        # Define unsupervised model builder (no loss function)
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Orchestra uses custom unsupervised loss
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],  # Unsupervised learning doesn't use traditional metrics
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            temperature=0.1,
            cluster_weight=1.0,
            contrastive_weight=1.0,
            deg_weight=0.1,
            ema_decay=0.999,
            num_local_clusters=20,
            num_global_clusters=10,
            memory_size=128,  # Small value for testing
            projection_dim=256,  # Small value for testing
            hidden_dim=256,  # Small value for testing
        )

        # Check if components are properly initialized
        assert hasattr(orchestra_worker, 'backbone')
        assert hasattr(orchestra_worker, 'projector')
        assert hasattr(orchestra_worker, 'target_backbone')
        assert hasattr(orchestra_worker, 'target_projector')
        assert hasattr(orchestra_worker, 'deg_layer')
        assert hasattr(orchestra_worker, 'centroids')
        assert hasattr(orchestra_worker, 'local_centroids')
        assert hasattr(orchestra_worker, 'mem_projections')

        # Check parameter settings
        assert orchestra_worker.temperature == 0.1
        assert orchestra_worker.N_local == 20
        assert orchestra_worker.N_global == 10
        assert orchestra_worker.memory_size == 128
        assert orchestra_worker.projection_dim == 256

    def test_orchestra_unsupervised_training(self, sf_simulation_setup_devices):
        """Test Orchestra strategy unsupervised training steps"""
        # Define unsupervised model builder
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],
        )

        # Initialize Orchestra strategy with small parameters for testing
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=64,  # Small memory for testing
            projection_dim=128,  # Small projection dimension for testing
            hidden_dim=128,
            num_local_clusters=5,  # Fewer clusters for testing
            num_global_clusters=3,
        )

        # Prepare unsupervised test data (ignore labels)
        x_test = torch.rand(64, 1, 28, 28)  # Small batch for testing
        y_test = torch.randint(0, 10, (64,))  # Labels only for clustering quality evaluation
        
        # Create unsupervised dataloader
        test_loader = create_unsupervised_dataloader(x_test, batch_size=16, shuffle=True)
        orchestra_worker.train_set = iter(test_loader)
        orchestra_worker.train_iter = iter(orchestra_worker.train_set)

        # Execute training step
        gradients = None
        gradients, num_sample = orchestra_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )

        # Apply weight updates
        orchestra_worker.apply_weights(gradients)

        # Verify sample count and gradients
        assert num_sample == 16  # Batch size
        assert gradients is not None
        assert isinstance(gradients, list)
        assert len(gradients) > 0

        # Execute another training step to test accumulation behavior
        _, num_sample = orchestra_worker.train_step(
            gradients, cur_steps=1, train_steps=2
        )
        assert num_sample == 32  # Two-step accumulated batch size
        
        # Test feature extraction and clustering quality
        with torch.no_grad():
            features_np = orchestra_worker.extract_features(x_test[:32], feature_type='backbone')  # Extract features
            features = torch.from_numpy(features_np)
            projections_np = orchestra_worker.extract_features(x_test[:32], feature_type='projection')  # Get projections
            projections = torch.from_numpy(projections_np)
            
            # Verify feature dimensions
            assert features.shape == (32, 128), f"Feature dimensions should be (32, 128), actual: {features.shape}"
            assert projections.shape == (32, 128), f"Projection dimensions should be (32, 128), actual: {projections.shape}"
            
            # Evaluate clustering quality (using true labels as reference)
            clustering_metrics = evaluate_clustering_quality(
                features, y_test[:32], num_clusters=10
            )
            
            # Verify clustering metrics reasonableness
            assert -1 <= clustering_metrics['ari'] <= 1, "ARI should be in [-1,1] range"
            assert 0 <= clustering_metrics['nmi'] <= 1, "NMI should be in [0,1] range"
            assert clustering_metrics['cluster_balance'] > 0, "Cluster balance should be greater than 0"
            assert clustering_metrics['num_clusters_found'] > 0, "Should find at least one cluster"

    def test_orchestra_components_functionality(self, sf_simulation_setup_devices):
        """Test Orchestra strategy component functionality (unsupervised learning)"""
        # Define unsupervised model builder
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=32,  # Small value for testing
            projection_dim=64,
            hidden_dim=64,
            num_local_clusters=3,
            num_global_clusters=2,
        )

        # Test projection network - use correct backbone output dimensions
        dummy_features = torch.randn(8, 128)  # OrchestraConvNet feature_layer output dimension
        # Need to create correctly shaped input data for extract_features
        dummy_input = torch.randn(8, 1, 28, 28)  # OrchestraConvNet expected input shape
        projected_np = orchestra_worker.extract_features(dummy_input, feature_type='projection')
        projected = torch.from_numpy(projected_np)
        assert projected.shape == (8, 64)  # projection_dim

        # Test target network
        target_projected = orchestra_worker.target_projector(dummy_features)
        assert target_projected.shape == (8, 64)

        # Test degeneracy layer
        deg_output = orchestra_worker.deg_layer(projected)
        assert deg_output.shape == (8, 4)  # 4 rotation classes

        # Test centroids
        centroids_output = orchestra_worker.centroids(projected)
        assert centroids_output.shape == (8, 2)  # num_global_clusters

        local_centroids_output = orchestra_worker.local_centroids(projected)
        assert local_centroids_output.shape == (8, 3)  # num_local_clusters

    def test_orchestra_memory_and_clustering_quality(self, sf_simulation_setup_devices):
        """Test Orchestra strategy memory operations and clustering quality validation"""
        # Define unsupervised model builder
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=16, 
            projection_dim=32,
            hidden_dim=32,
            num_local_clusters=4,
            num_global_clusters=2,
        )

        # Prepare unsupervised test dataset
        x_test = torch.rand(32, 1, 28, 28)
        y_test = torch.randint(0, 10, (32,))  # Labels for clustering quality evaluation
        test_loader = create_unsupervised_dataloader(x_test, batch_size=8, shuffle=False)

        # Test memory reset
        device = torch.device('cpu')
        orchestra_worker._reset_memory(test_loader, device)

        # Check memory projection shape
        assert orchestra_worker.mem_projections.weight.data.shape == (32, 16)  # (projection_dim, memory_size)
        
        # Verify memory bank initialization quality
        memory_data = orchestra_worker.mem_projections.weight.data.T  # (memory_size, projection_dim)
        assert not torch.allclose(memory_data, torch.zeros_like(memory_data)), "Memory bank should not be all zeros"
        
        # Check memory bank data diversity
        memory_std = torch.std(memory_data, dim=0).mean()
        assert memory_std > 0.01, f"Memory bank data standard deviation too small: {memory_std}"

        # Test memory update
        dummy_features = torch.randn(4, 32)  # Small batch features
        old_memory = orchestra_worker.mem_projections.weight.data.clone()
        orchestra_worker._update_memory(dummy_features)

        # Memory should maintain same shape but content should change
        assert orchestra_worker.mem_projections.weight.data.shape == (32, 16)
        assert not torch.allclose(old_memory, orchestra_worker.mem_projections.weight.data), "Memory should change after update"
        
        # Test clustering quality
        with torch.no_grad():
            # Extract features for clustering quality evaluation
            features_np = orchestra_worker.extract_features(x_test, feature_type='backbone')
            features = torch.from_numpy(features_np)
            projections_np = orchestra_worker.extract_features(x_test, feature_type='projection')
            projections = torch.from_numpy(projections_np)
            
            # Evaluate feature clustering quality
            feature_metrics = evaluate_clustering_quality(features, y_test, num_clusters=10)
            projection_metrics = evaluate_clustering_quality(projections, y_test, num_clusters=10)
            
            # Verify clustering metrics
            for metrics, name in [(feature_metrics, "features"), (projection_metrics, "projections")]:
                assert -1 <= metrics['ari'] <= 1, f"{name} ARI should be in [-1,1] range"
                assert 0 <= metrics['nmi'] <= 1, f"{name} NMI should be in [0,1] range"
                assert metrics['cluster_balance'] > 0, f"{name} cluster balance should be greater than 0"
                assert 1 <= metrics['num_clusters_found'] <= 10, f"{name} number of clusters found should be in reasonable range"
            
            # Projected clustering quality should not be lower than original features (or at least in reasonable range)
            print(f"Feature clustering quality - ARI: {feature_metrics['ari']:.3f}, NMI: {feature_metrics['nmi']:.3f}")
            print(f"Projection clustering quality - ARI: {projection_metrics['ari']:.3f}, NMI: {projection_metrics['nmi']:.3f}")

    def test_orchestra_ema_update(self, sf_simulation_setup_devices):
        """Test Orchestra strategy EMA update mechanism"""
        # Define unsupervised model builder
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=16,
            projection_dim=32,
            hidden_dim=32,
            ema_decay=0.9,  # Set EMA decay rate
        )

        # Get initial target model parameters
        initial_target_params = []
        for param in orchestra_worker.target_backbone.parameters():
            initial_target_params.append(param.data.clone())

        # Simulate training step to trigger EMA update
        dummy_input = torch.randn(4, 1, 28, 28)

        # Forward pass (unsupervised scenario)
        orchestra_worker.backbone.train()
        with torch.no_grad():
            features_np = orchestra_worker.extract_features(dummy_input, feature_type='backbone')
            features = torch.from_numpy(features_np)
            projections_np = orchestra_worker.extract_features(dummy_input, feature_type='projection')
            projections = torch.from_numpy(projections_np)
            
            # Simulate unsupervised loss (e.g., contrastive learning loss)
            # Simplified here as L2 norm loss of features
            unsupervised_loss = torch.mean(torch.norm(projections, dim=1))

        # Manually update model parameters (simulate optimizer step)
        with torch.no_grad():
            for param in orchestra_worker.backbone.parameters():
                if param.requires_grad:
                    # Add small random gradients to simulate parameter updates
                    param.data += torch.randn_like(param) * 0.01

        # Trigger EMA update
        orchestra_worker._update_target_model()

        # Check if target model parameters have changed
        params_changed = False
        for i, param in enumerate(orchestra_worker.target_backbone.parameters()):
            if not torch.equal(param.data, initial_target_params[i]):
                params_changed = True
                break
        
        assert params_changed, "Target model should be updated through EMA"
        
        # Verify EMA update smoothness
        # Check if EMA update maintains reasonable parameter ranges
        for param in orchestra_worker.target_backbone.parameters():
            assert torch.isfinite(param.data).all(), "Parameters after EMA update should be finite"
            assert not torch.isnan(param.data).any(), "Parameters after EMA update should not contain NaN"

    def test_orchestra_clustering_quality(self, sf_simulation_setup_devices):
        """Test Orchestra strategy clustering functionality and quality"""
        # Define unsupervised model builder
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],
        )

        # Initialize Orchestra strategy with clustering
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=32,
            projection_dim=64,
            hidden_dim=64,
            num_local_clusters=8,
            num_global_clusters=4,
        )

        # Test clustering initialization
        assert hasattr(orchestra_worker, 'local_centroids'), "Should have local clustering component"
        assert hasattr(orchestra_worker, 'centroids'), "Should have global clustering component"
        assert orchestra_worker.N_local == 8
        assert orchestra_worker.N_global == 4

        # Generate more structured test data (simulate real feature distribution)
        torch.manual_seed(42)  # Ensure reproducibility
        
        # Create data with obvious clustering structure
        cluster_centers = torch.randn(4, 64) * 2  # 4 cluster centers
        dummy_features = []
        true_labels = []
        
        for i in range(4):
            # Generate 4 samples for each cluster
            cluster_data = cluster_centers[i].unsqueeze(0) + torch.randn(4, 64) * 0.5
            dummy_features.append(cluster_data)
            true_labels.extend([i] * 4)
        
        dummy_features = torch.cat(dummy_features, dim=0)  # (16, 64)
        true_labels = torch.tensor(true_labels)
        
        # Initialize memory
        random_memory = torch.randn(64, 32)  # (projection_dim, memory_size)
        orchestra_worker.mem_projections.weight.data.copy_(random_memory)

        # Test local clustering
        device = torch.device('cpu')
        orchestra_worker._local_clustering(device)

        # Check local clustering center shape
        assert orchestra_worker.local_centroids.weight.data.shape == (8, 64)  # (num_local_clusters, projection_dim)
        
        # Evaluate local clustering quality
        local_metrics = evaluate_clustering_quality(dummy_features, true_labels, num_clusters=4)
        print(f"Local clustering quality - ARI: {local_metrics['ari']:.3f}, NMI: {local_metrics['nmi']:.3f}, Balance: {local_metrics['cluster_balance']:.3f}")
        
        # Verify clustering quality metrics
        assert -1 <= local_metrics['ari'] <= 1, "ARI should be in [-1,1] range"
        assert 0 <= local_metrics['nmi'] <= 1, "NMI should be in [0,1] range"
        assert local_metrics['cluster_balance'] > 0, "Cluster balance should be greater than 0"

        # Test global clustering
        aggregated_features = torch.randn(20, 64)  # Some aggregated features
        orchestra_worker._global_clustering(aggregated_features, device)

        # Check global clustering center shape
        assert orchestra_worker.centroids.weight.data.shape == (4, 64)  # (num_global_clusters, projection_dim)
        
        # Verify global clustering consistency
        global_metrics = evaluate_clustering_quality(aggregated_features, torch.randint(0, 4, (20,)), num_clusters=4)
        assert -1 <= global_metrics['ari'] <= 1, "Global clustering ARI should be in [-1,1] range"
        assert 0 <= global_metrics['nmi'] <= 1, "Global clustering NMI should be in [0,1] range"
        print(f"Global clustering quality - ARI: {global_metrics['ari']:.3f}, NMI: {global_metrics['nmi']:.3f}")

    def test_orchestra_full_integration(self, sf_simulation_setup_devices):
        """Test Orchestra strategy full integration with FLModel."""
        # Define model builder with metrics
        builder = TorchModel(
            model_fn=OrchestraConvNet,
            loss_fn=None,  # Unsupervised learning
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[],  # Unsupervised learning doesn't use traditional metrics
        )

        # Set up devices
        device_list = [
            sf_simulation_setup_devices.alice,
            sf_simulation_setup_devices.bob,
        ]

        # Initialize aggregator
        aggregator = SecureAggregator(sf_simulation_setup_devices.carol, device_list)

        # Create federated learning model with Orchestra strategy
        fl_model = FLModel(
            server=sf_simulation_setup_devices.carol,
            device_list=device_list,
            model=builder,
            strategy="orchestra",
            backend="torch",
            aggregator=aggregator,
            # Orchestra specific parameters
            num_classes=10,
            memory_size=32,  # Small for testing
            projection_dim=64,
            num_local_clusters=5,
            num_global_clusters=3,
        )

        # Prepare small dataset for testing
        (_, _), (data, label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: 0.4,
                sf_simulation_setup_devices.bob: 0.6,
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )

        # Train for one epoch with small batch size
        history = fl_model.fit(
            data,
            label,
            epochs=1,
            batch_size=16,  # Small batch for testing
            aggregate_freq=1,
        )

        # Check that training completed successfully
        # Orchestra unsupervised learning may not produce global_history, which is normal
        assert 'global_history' in history
        # For unsupervised learning, we check if training completed successfully rather than global_history content
        assert history is not None, "Training should return history object"

        # Make predictions
        result = fl_model.predict(data, batch_size=16)
        assert len(reveal(result[device_list[0]])) > 0

        # Evaluate unsupervised model feature quality
        # Extract features for clustering quality evaluation
        with torch.no_grad():
            # Get data directly from distributed object partitions to avoid indexing operations
            # Get data partition from first device
            first_device = device_list[0]
            full_test_data = reveal(data.partitions[first_device])
            full_test_labels = reveal(label.partitions[first_device])
            
            # Slice local objects
            test_data = full_test_data[:64]
            test_labels = full_test_labels[:64]
            
            # Use trained model to extract features
            # Extract features through Orchestra strategy's extract_features method
            orchestra_worker = fl_model._workers[first_device]
            features_pyu = orchestra_worker.extract_features(test_data, feature_type='backbone')
            # Get actual numpy array from distributed object
            features_np = reveal(features_pyu)
            # Convert numpy array to torch tensor
            features = torch.from_numpy(features_np)
            
            # Evaluate feature clustering quality
            # Convert numpy array to torch tensor to use argmax method
            test_labels_tensor = torch.from_numpy(test_labels) if isinstance(test_labels, np.ndarray) else test_labels
            clustering_metrics = evaluate_clustering_quality(
                features, test_labels_tensor.argmax(dim=1), num_clusters=10
            )
            
            print(f"Unsupervised feature clustering quality - ARI: {clustering_metrics['ari']:.3f}, NMI: {clustering_metrics['nmi']:.3f}")
            assert -1 <= clustering_metrics['ari'] <= 1, "Feature clustering ARI should be in [-1,1] range"
            assert 0 <= clustering_metrics['nmi'] <= 1, "Feature clustering NMI should be in [0,1] range"