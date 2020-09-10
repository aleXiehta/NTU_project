# NTU_project
Pressure measuring

## Requirements
pytorch 1.4.0 (conda)
torchvision 0.5.0 (conda)
pretrainmodels 0.7.4 (pip)
scikit-image 0.17.2 (pip)

## Usage
```python
from predict import Predictor
model = Predictor('/path/to/checkpoint')
model.run('/path/to/image.jpg')
