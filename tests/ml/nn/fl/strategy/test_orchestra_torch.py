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
from torchmetrics import Accuracy, Precision

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


class ConvNet(BaseModule):
    """Small ConvNet for MNIST used in Orchestra testing."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3))
        x = x.view(-1, self.fc_in_dim)
        x = F.relu(self.fc(x))
        return x


class TestOrchestraStrategy:
    """Test cases for Orchestra unsupervised federated learning strategy."""

    def test_orchestra_initialization(self, sf_simulation_setup_devices):
        """Test Orchestra strategy initialization."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
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
            memory_size=128,  # Reduced for testing
            projection_dim=256,  # Reduced for testing
            hidden_dim=256,  # Reduced for testing
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

    def test_orchestra_local_step(self, sf_simulation_setup_devices):
        """Test Orchestra strategy local training step."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

        # Initialize Orchestra strategy with smaller parameters for testing
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=64,  # Small memory for testing
            projection_dim=128,  # Small projection dim for testing
            hidden_dim=128,
            num_local_clusters=5,  # Fewer clusters for testing
            num_global_clusters=3,
        )

        # Prepare test dataset
        x_test = torch.rand(64, 1, 28, 28)  # Small batch for testing
        y_test = torch.randint(0, 10, (64,))
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=16, shuffle=True
        )
        orchestra_worker.train_set = iter(test_loader)
        orchestra_worker.train_iter = iter(orchestra_worker.train_set)

        # Perform a training step
        gradients = None
        gradients, num_sample = orchestra_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )

        # Apply weights update
        orchestra_worker.apply_weights(gradients)

        # Assert the sample number and gradients
        assert num_sample == 16  # Batch size
        assert gradients is not None
        assert isinstance(gradients, list)
        assert len(gradients) > 0

        # Perform another training step to test cumulative behavior
        _, num_sample = orchestra_worker.train_step(
            gradients, cur_steps=1, train_steps=2
        )
        assert num_sample == 32  # Cumulative batch size over two steps

    def test_orchestra_components_functionality(self, sf_simulation_setup_devices):
        """Test Orchestra strategy component functionality."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=32,  # Very small for testing
            projection_dim=64,
            hidden_dim=64,
            num_local_clusters=3,
            num_global_clusters=2,
        )

        # Test projection network
        dummy_features = torch.randn(8, 192)  # Backbone output dimension
        projected = orchestra_worker.projector(dummy_features)
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

    def test_orchestra_memory_operations(self, sf_simulation_setup_devices):
        """Test Orchestra strategy memory operations."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            num_classes=10,
            memory_size=16,  # Very small for testing
            projection_dim=32,
            hidden_dim=32,
        )

        # Prepare small test dataset for memory reset
        x_test = torch.rand(32, 1, 28, 28)
        y_test = torch.randint(0, 10, (32,))
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=8, shuffle=False
        )

        # Test memory reset
        device = torch.device('cpu')
        orchestra_worker._reset_memory(test_loader, device)

        # Check memory projections shape
        assert orchestra_worker.mem_projections.weight.data.shape == (32, 16)  # (projection_dim, memory_size)

        # Test memory update
        dummy_features = torch.randn(4, 32)  # Small batch of features
        orchestra_worker._update_memory(dummy_features)

        # Memory should still have the same shape
        assert orchestra_worker.mem_projections.weight.data.shape == (32, 16)

    def test_orchestra_ema_update(self, sf_simulation_setup_devices):
        """Test Orchestra strategy EMA target model update."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            ema_decay=0.9,  # Lower decay for more noticeable changes
        )

        # Get initial target model parameters
        initial_target_params = []
        for param in orchestra_worker.target_backbone.parameters():
            initial_target_params.append(param.data.clone())

        # Modify online model parameters
        for param in orchestra_worker.backbone.parameters():
            param.data += 0.1  # Add some change

        # Update target model
        orchestra_worker._update_target_model()

        # Check that target model parameters have changed
        for i, param in enumerate(orchestra_worker.target_backbone.parameters()):
            assert not torch.equal(param.data, initial_target_params[i])

    def test_orchestra_clustering(self, sf_simulation_setup_devices):
        """Test Orchestra strategy clustering operations."""
        # Define model builder
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

        # Initialize Orchestra strategy
        orchestra_worker = OrchestraStrategy(
            builder_base=builder,
            memory_size=16,
            projection_dim=32,
            num_local_clusters=4,
            num_global_clusters=2,
        )

        # Initialize memory with random data
        random_memory = torch.randn(32, 16)  # (projection_dim, memory_size)
        orchestra_worker.mem_projections.weight.data.copy_(random_memory)

        # Test local clustering
        device = torch.device('cpu')
        orchestra_worker._local_clustering(device)

        # Check local centroids shape
        assert orchestra_worker.local_centroids.weight.data.shape == (4, 32)  # (num_local_clusters, projection_dim)

        # Test global clustering
        aggregated_features = torch.randn(20, 32)  # Some aggregated features
        orchestra_worker._global_clustering(aggregated_features, device)

        # Check global centroids shape
        assert orchestra_worker.centroids.weight.data.shape == (2, 32)  # (num_global_clusters, projection_dim)

    def test_orchestra_full_integration(self, sf_simulation_setup_devices):
        """Test Orchestra strategy full integration with FLModel."""
        # Define model builder with metrics
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
            ],
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
            validation_data=(data, label),
            epochs=1,
            batch_size=16,  # Small batch for testing
            aggregate_freq=1,
        )

        # Check that training completed successfully
        assert 'global_history' in history
        assert len(history['global_history']) > 0

        # Make predictions
        result = fl_model.predict(data, batch_size=16)
        assert len(reveal(result[device_list[0]])) > 0

        # Evaluate model
        global_metric, _ = fl_model.evaluate(
            data, label, batch_size=16, random_seed=1234
        )
        assert len(global_metric) > 0
        assert global_metric[0].result().numpy() >= 0.0  # Accuracy should be non-negative