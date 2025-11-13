# Medical Image Analysis Starter

A Python template for medical image analysis tasks, including feature extraction, model training, and evaluation. This template provides a foundation for working with medical imaging data, particularly focused on radiomics features and machine learning.

## Features

- Sample Data Generation: Create synthetic medical imaging features for testing
- Feature Engineering: Extract and analyze radiomic features
- Model Training: Built-in Random Forest classifier with feature importance
- Evaluation Metrics: Classification reports and visualizations
- Extensible Design: Easy to modify for specific medical imaging tasks

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- pydicom (for DICOM file support)

## Installation

1. Prerequisites:
   - Ensure you have Python 3.7 or later installed
   - Verify Python installation:
          python --version
     
   - If needed, download Python from [python.org](https://www.python.org/downloads/)

2. Set up a virtual environment (recommended):
      # Create a virtual environment
   python -m venv medical_imaging_env
   
   # Activate the environment
   # Windows:
   .\medical_imaging_env\Scripts\activate
   # macOS/Linux:
   # source medical_imaging_env/bin/activate
   

3. Install required packages:
      # Core requirements
   pip install numpy pandas matplotlib scikit-learn seaborn pydicom
   
   # For Jupyter notebook support (optional)
   pip install notebook
   
   # For additional medical imaging processing (optional)
   # pip install SimpleITK nibabel
   

4. Verify installation:
      python -c "import pydicom; import seaborn as sns; print('Medical imaging packages installed successfully!')"
   

## Usage

from medical_image_analyzer import MedicalImageAnalyzer

# Initialize the analyzer
analyzer = MedicalImageAnalyzer()

# Load and preprocess data (uses sample data if no path provided)
X, y = analyzer.load_and_preprocess("path/to/your/dicom/folder")

# Train the model
X_test, y_test = analyzer.train_model(X, y)

# Evaluate the model
analyzer.evaluate_model(X_test, y_test)

## Features Overview

The template works with the following radiomic features:

1. mean_intensity: Average intensity of the region
2. std_intensity: Standard deviation of intensity values
3. skewness: Measure of intensity distribution asymmetry
4. kurtosis: Tailedness of the intensity distribution
5. entropy: Randomness in the intensity values
6. contrast: Local variations in the image
7. correlation: Linear dependency of gray levels
8. energy: Sum of squared elements
9. homogeneity: Closeness of element distributions
10. area: Size of the region of interest

## Model Evaluation

The template provides:
- Classification report with precision, recall, and F1-score
- Feature importance visualization
- Easy integration with other scikit-learn models

## Extending the Template

To add custom features or models:

class CustomImageAnalyzer(MedicalImageAnalyzer):
    def __init__(self):
        super().__init__()
        
    def extract_custom_features(self, image_data):
        """Add custom feature extraction logic"""
        # Your custom feature extraction code here
        pass

## Troubleshooting

- DICOM Support: Ensure pydicom is installed for DICOM file support
- Memory Issues: For large datasets, consider processing in batches
- Visualization Backend: If using in a non-GUI environment, set:
    import matplotlib
  matplotlib.use('Agg')  # Use non-interactive backend
  

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
