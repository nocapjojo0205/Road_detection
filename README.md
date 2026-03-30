# Road Condition Detection System

A fine-grained road surface condition detection system for intelligent driving, supporting multi-class recognition and dynamic safety warning.

## Features
- Six-category road condition recognition (dry, wet, icy, snowy, blowing snow, melting snow)
- Improved ResNet34 with CBAM attention mechanism
- Vehicle dynamics-based dynamic safe distance model
- Three-level hierarchical early warning strategy
- PyQt graphical user interface for real-time detection and warning

## Tech Stack
- PyTorch
- ResNet34-CBAM
- PyQt5
- OpenCV

## Dataset
https://www.ncdc.ac.cn/portal/

## Experimental Results
- Average accuracy: 89.50%
- Significant improvement on transitional road states (blowing snow, melting snow)

## Quick Start
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the GUI system: python main_ui.py
4. Train the model from scratch python scripts/train.py --config configs/base_config.yaml

## License
MIT

## Contribution
Feel free to submit issues and pull requests to improve this project
