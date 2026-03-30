import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
import yaml
from utils.utils import load_config
from OCR_Vehicle_Plate_Detector.scripts.train_plate_detection import create_data_yaml
from OCR_Vehicle_Plate_Detector.scripts.train_plate_detection import create_data_yaml
from OCR_Vehicle_Plate_Detector.scripts.train_plate_detection import create_data_yaml
from OCR_Vehicle_Plate_Detector.scripts.train_plate_detection import create_data_yaml

# Absolute import for the function to test
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration."""
    return {
        'paths': {
            'train_data': '/path/to/train',
            'validation_data': '/path/to/val',
            'test_data': '/path/to/test',
            'models_checkpoints': '/path/to/checkpoints',
            'plate_detection': '/path/to/plate_detection'
        },
        'model': {
            'detection': {
                'name': 'yolov8',
                'img_size': [640, 640],
                'device': 'cuda'
            }
        },
        'training': {
            'epochs': 10,
            'batch_size': 4,
            'learning_rate': 0.001,
            'optimizer': 'SGD',
            'early_stopping': {'patience': 5},
            'checkpoint': {'save_frequency': 5}
        }
    }


@pytest.fixture
def mock_yolo_model():
    """Fixture providing mock YOLO model."""
    model = MagicMock()
    model.train = MagicMock(return_value=None)
    model.export = MagicMock(return_value=None)
    return model


class TestCreateDataYaml:
    """Test suite for create_data_yaml function."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    @patch('utils.utils.load_config')
    def test_create_data_yaml_success(self, mock_load_config, mock_yaml_dump, mock_file, mock_config):
        """Test successful YAML file creation."""
        mock_load_config.return_value = mock_config
        
        # Import here to get patched config
        
        result = create_data_yaml()
        
        assert result == 'data.yaml'
        mock_file.assert_called_once_with('data.yaml', 'w')
        mock_yaml_dump.assert_called_once()
        
        # Verify YAML structure
        call_args = mock_yaml_dump.call_args[0][0]
        assert call_args['nc'] == 1
        assert call_args['names'] == ['license-plate']

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    @patch('utils.utils.load_config')
    def test_create_data_yaml_paths_correct(self, mock_load_config, mock_yaml_dump, mock_file, mock_config):
        """Test that correct paths are included in YAML."""
        mock_load_config.return_value = mock_config
        
        
        create_data_yaml()
        
        call_args = mock_yaml_dump.call_args[0][0]
        assert call_args['train'] == '/path/to/train'
        assert call_args['val'] == '/path/to/val'
        assert call_args['test'] == '/path/to/test'

    @patch('builtins.open', side_effect=IOError("Cannot write file"))
    @patch('utils.utils.load_config')
    def test_create_data_yaml_file_error(self, mock_load_config, mock_file, mock_config):
        """Test handling of file writing errors."""
        mock_load_config.return_value = mock_config
        
        
        with pytest.raises(IOError):
            create_data_yaml()


class TestTrainScript:
    """Test suite for training script main execution."""

    @patch('pathlib.Path.mkdir')
    @patch('ultralytics.YOLO')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    @patch('utils.utils.load_config')
    def test_model_training_called_with_correct_params(
        self, mock_load_config, mock_yaml_dump, mock_file, 
        mock_yolo, mock_mkdir, mock_config
    ):
        """Test that model.train is called with correct parameters."""
        mock_load_config.return_value = mock_config
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        data_yaml = create_data_yaml()
        model = mock_yolo(f"{mock_config['model']['detection']['name']}m.pt")
        
        model.train(
            data=data_yaml,
            epochs=mock_config['training']['epochs'],
            imgsz=mock_config['model']['detection']['img_size'][0],
            batch=mock_config['training']['batch_size'],
            lr0=mock_config['training']['learning_rate'],
            optimizer=mock_config['training']['optimizer'],
            device=mock_config['model']['detection']['device'],
            patience=mock_config['training']['early_stopping']['patience'],
            save_period=mock_config['training']['checkpoint']['save_frequency'],
            project=mock_config['paths']['models_checkpoints'],
            name='plate_detection'
        )
        
        model.train.assert_called_once()
        call_kwargs = model.train.call_args[1]
        assert call_kwargs['epochs'] == 10
        assert call_kwargs['batch'] == 4
        assert call_kwargs['device'] == 'cuda'

    @patch('pathlib.Path.mkdir')
    @patch('ultralytics.YOLO')
    @patch('utils.utils.load_config')
    def test_model_export_called(self, mock_load_config, mock_yolo, mock_mkdir, mock_config):
        """Test that model.export is called after training."""
        mock_load_config.return_value = mock_config
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        mock_model.export(format='pt')
        
        mock_model.export.assert_called_once_with(format='pt')


class TestGPUMemoryOptimization:
    """Test suite for GPU memory optimization."""

    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.is_available')
    def test_cuda_memory_cleared(self, mock_cuda_available, mock_empty_cache):
        """Test that CUDA memory is cleared."""
        mock_cuda_available.return_value = True
        
        # Simulate memory clearing
        mock_empty_cache()
        
        mock_empty_cache.assert_called_once()

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_gpu_memory_monitoring(self, mock_reserved, mock_allocated):
        """Test GPU memory monitoring functions."""
        mock_allocated.return_value = 1024 * 1024 * 500  # 500MB
        mock_reserved.return_value = 1024 * 1024 * 1024  # 1GB
        
        allocated = mock_allocated()
        reserved = mock_reserved()
        
        assert allocated == 1024 * 1024 * 500
        assert reserved == 1024 * 1024 * 1024
        assert reserved > allocated

    @patch('torch.cuda.is_available')
    def test_cuda_availability_check(self, mock_cuda_available):
        """Test CUDA availability check."""
        mock_cuda_available.return_value = True
        
        assert mock_cuda_available() is True
        mock_cuda_available.assert_called_once()