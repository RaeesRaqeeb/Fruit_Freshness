# Fruit Inspection System - Freshness Classification

## Overview

Fruit Inspection System is an artificial neural network-based solution for automated fruit freshness classification. The system uses convolutional neural networks (CNN) to classify fruits into fresh and rotten categories across 9 different fruit types. This is a course project for ANN (Artificial Neural Networks) that demonstrates practical application of deep learning in agricultural quality control.

## Project Scope

The system classifies 18 different fruit conditions:
- Fresh variants: Apples, Banana, Bitter Gourd, Capsicum, Cucumber, Okra, Oranges, Potato, Tomato
- Rotten variants: Apples, Banana, Bitter Gourd, Capsicum, Cucumber, Okra, Oranges, Potato, Tomato

The application provides both a REST API and a web interface for easy integration and user interaction.

## Team Members

- Raqeeb Raees (Lead)
- Asaf Khan
- Naeem Khan

## Features

- CNN-based fruit freshness classification with 18 classes
- FastAPI REST API endpoint for model inference
- Web-based interface for real-time fruit scanning
- Support for multiple model formats (Keras and PyTorch)
- Docker containerization for easy deployment
- Confidence threshold configuration for prediction reliability
- Image upload and processing capabilities

## System Architecture

The system consists of two main components:

### 1. Model Training and Inference Engine
   - Trained CNN models using Keras 
   - classification (Fresh/Rotten) across 9 fruit types
   - Models trained on fruit freshness dataset

### 2. Web Application
   - FastAPI backend server
   - HTML5 frontend with image upload capability
   - Real-time prediction display
   - Responsive design for cross-device compatibility

## Technologies Used

- Python 3.8+
- TensorFlow/Keras
- FastAPI
- NumPy

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository
   git clone <repository-url>
   cd Fruit_Freshness

2. Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the application
   uvicorn main:app --reload

5. Access the web interface
   Open http://localhost:8000 in your browser

### Docker Setup

### Endpoint: POST /predict

```bash
docker build -t fruit-inspection-system .
docker run -p 8000:8000 fruit-inspection-system
```

#### Access the application at http://localhost:8000

## API Usage

Endpoint: POST /predict

## Request:
- Method: POST
- Content-Type: multipart/form-data
- Parameter: file (image file)

## Response:
{
  "prediction": "freshapples",
  "confidence": 0.95,
   "status": "Fresh"
}

## Web Interface

The web interface provides a user-friendly way to:
- Upload fruit imagesand PyTorch (.pth format)
- Display confidence scores
- Show predicted fruit freshness status

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: RGB images (standardized dimensions)
- Output: 18 class predictions with confidence scores
- Confidence Threshold: Configurable (default: 0.85)
- Model Files: Keras (.keras format) 

## Training Dataset

The models were trained using a comprehensive fruit freshness dataset. For detailed training methodology and dataset information, refer to the Kaggle notebook:

Kaggle Notebook: https://www.kaggle.com/code/raqeeb24/fruit-classifier/

## Production Readiness - Future Enhancements

To make this system production-ready, the following improvements and enhancements are planned:

### 1. Pre-trained Model Integration
   - Leverage transfer learning with pre-trained models (ResNet50, MobileNetV2, EfficientNet)
   - Reduce training time and improve accuracy with ImageNet-pretrained weights
   - Fine-tune pre-trained models on fruit freshness dataset
   - Implement ensemble methods combining multiple pre-trained architectures

### 2. Transfer Learning Implementation
   - Adapt state-of-the-art vision models to fruit classification task
   - Utilize feature extraction from larger datasets
   - Implement domain adaptation techniques
   - Create lightweight models for edge deployment

### 3. Advanced Detection and Classification
   - Implement YOLO (You Only Look Once) v8 for real-time fruit detection
   - Enable multi-fruit detection in single images
   - Combine detection and classification pipeline
   - Real-time processing on video streams
   - Multi-fruit batch processing capability

### 4. Performance Optimization
   - Model quantization for reduced inference time
   - ONNX model export for cross-platform compatibility
   - GPU acceleration support
   - Edge device deployment (Raspberry Pi, NVIDIA Jetson)

### 5. Scalability and Monitoring
   - Distributed inference pipeline
   - Model versioning and A/B testing
   - Real-time performance monitoring and logging
   - Database integration for prediction history
   - Automated retraining pipeline

### 6. Robustness Enhancement
   - Data augmentation for edge cases
   - Adversarial testing and validation
   - Multi-angle fruit inspection support
   - Lighting and background invariance
   - Seasonal variation handling

### 7. Production Deployment
   - Kubernetes orchestration
   - Load balancing for high-throughput scenarios
   - CI/CD pipeline integration
   - Automated model updates
   - Health checks and graceful degradation

## File Structure

Fruit_Freshness/
- main.py                 # FastAPI application server
- inference.py            # Model inference logic
- Fruit_freshness.keras   # Trained Keras model
- requirements.txt        # Python dependencies
- Dockerfile              # Container configuration
- README.md               # This file

## Configuration

### Environment Variables:

- OTHER_LABEL: Label for unrecognized fruits (default: 'Others')
- UNKNOWN_STATUS: Status label for unknown predictions (default: 'Unknown')
- KERAS_REJECT_THRESHOLD: Confidence threshold for predictions (default: 0.60)

## Contributing

This project is a course submission for the ANN (Artificial Neural Networks) course. Contributions from team members are welcome for enhancements and bug fixes.

## Future Development and Collaboration

We welcome contributions and feedback for:
- Model accuracy improvements
- Performance optimization
- User interface enhancements
- Documentation updates
- Real-world deployment scenarios

## Testing and Validation

The system has been tested with various fruit images under different lighting conditions and angles. For comprehensive testing and performance metrics, refer to the course project documentation.

## License

This project is developed as part of an academic course project. Please refer to the project documentation for licensing information.

## Acknowledgments

- Course instructors and mentors
- Kaggle dataset contributors
- Open-source community for frameworks and tools

## Contact and Support

For issues, questions, or suggestions, please contact the project lead:
- Raqeeb Raees

For more information and detailed analysis, visit the Kaggle notebook:
https://www.kaggle.com/code/raqeeb24/fruit-classifier/
