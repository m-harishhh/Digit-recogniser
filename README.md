# MNIST Digit Classification using CNN and Random Forest

This project uses a ensembled approach to classify the MNIST dataset using a Convolutional Neural Network (CNN) for feature extraction and a Random Forest classifier for final predictions.

## Project Structure

```
├── code.ipynb          # Single file containing the entire project
├── README.md                # Project README file
├── submission.csv           # Prediction results for submission
```

## Overview

This project leverages CNNs to extract features from the MNIST dataset and a Random Forest classifier to predict the digit labels. The steps include:

1. **Data Preprocessing**: Reshaping and normalizing the dataset.
2. **CNN Model**: Building a CNN to extract features from the MNIST images.
3. **Random Forest Model**: Using the CNN features to train a Random Forest classifier.
4. **Prediction**: Making predictions on the test dataset and saving the results for submission.

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits. Each image is a 28x28 grayscale image, and the task is to predict the digit (0-9) that each image represents.

Download the dataset from [Kaggle - MNIST Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data).

## Requirements

Install the required dependencies using:

```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib
```

### Dependencies:

- Python 3.8+
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## How to Run the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/mnist-cnn-rf.git
   cd mnist-cnn-rf
   ```

2. **Download the dataset** and place it in the appropriate directory.

3. **Run the script** to train the models and generate predictions:

   ```bash
   python code.ipynb
   ```

   This will load the MNIST dataset, train the CNN model, extract features, and train the Random Forest classifier. Finally, predictions will be generated and saved in `submission.csv`.

## Results

- **Training Accuracy**: `99.97%`
- **Test Accuracy**: `99.03%` (Kaggle Leaderboard Score: 381/1378)

## Future Improvements

- **Hyperparameter Tuning**: Experiment with different hyperparameters for both CNN and Random Forest to improve accuracy.
- **Data Augmentation**: Introduce data augmentation techniques to improve model generalization.
- **Ensemble Methods**: Explore other ensemble methods like Gradient Boosting or XGBoost for improved classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [Kaggle](https://www.kaggle.com/) for the MNIST dataset and community contributions.
- TensorFlow and Keras for making deep learning frameworks accessible.

---

