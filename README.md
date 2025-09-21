# Deep Learning and Neural Network Experiment

## Experiment 1: Deep Neural Network Application
### Content
- **Feedforward Propagation Model**: Build a DNN model with an input layer and two hidden layers, implement the prediction function `predict.m` to achieve an expected accuracy of ~97.5%.
- **Backpropagation Model**: Complete the cost function `nnCostFunction.m` for both non-regularized and regularized cases, and verify the backpropagation algorithm through gradient calculation and checking.
- **Model Training and Validation**: Train the model using provided parameters (Theta1 and Theta2), and observe the impact of different training iterations and regularization coefficients on accuracy.
- **Activation Function Comparison**: Analyze the impact of different activation functions (sigmoid, ReLU, Softmax, tanh) on model performance.
### Results
- The feedforward model achieved an accuracy of 97.575%.
- The backpropagation model achieved an accuracy of 96.775%.
- Adjusting training iterations and the regularization coefficient affected accuracy trends.
- The sigmoid function outperformed other activation functions.

## Experiment 2: Convolutional Neural Network Application
### Content
- **Environment Configuration**: Set up Python and TensorFlow, and use Google Colab for efficient model training.
- **Data Preprocessing**: Load the CIFAR-10 dataset, normalize image data to [0, 1], and one-hot encode labels.
- **CNN Construction**: Build a CNN with Keras' Sequential model, including convolutional, max-pooling, Dropout, Flatten, and dense layers.
- **Model Compilation and Training**: Use the Adam optimizer with a learning rate of 0.001, train for 50 epochs with a batch size of 32, and validate on a held-out validation set.
- **Model Evaluation and Testing**: Plot loss and accuracy curves, and evaluate the model on the test set (accuracy: 75.08%).
- **Model Optimization**: Experiment with different learning rates, activation functions, loss functions, network structures, and the VGG-16 network.
### Results
- The baseline CNN achieved 75.08% test accuracy.
- Lowering the learning rate improved accuracy to 81.86%.
- The VGG-16 network achieved 85.49% accuracy, highlighting the advantage of complex structures.

## Experiment 3: Support Vector Machine
### Content
- **Gaussian Kernel Implementation**: Implement the Gaussian kernel to calculate similarity between samples.
- **Linear SVM**: Train linear SVM models with C=1 and C=100, and visualize decision boundaries.
- **Non-linear SVM**: Train non-linear SVM using the Gaussian kernel and visualize the decision boundary.
- **Parameter Optimization**: Search for optimal C and σ parameters using training and validation sets.
- **Spam Email Classification**: Preprocess and extract features from emails, train an SVM model, and evaluate its performance (training accuracy: 99.8%, test accuracy: 98.9%).
### Results
- Linear SVM showed significant decision boundary changes with different C values, with higher C leading to closer data fitting but potential overfitting.
- Non-linear SVM with the Gaussian kernel effectively handled non-linear data, with sigma significantly impacting classification.
- Optimal parameters found: C=0.2 and σ=0.01, with low validation error.
- The spam classification model showed high accuracy, with certain words having high weights for spam identification.

  ## Running Steps
1. Experiment 1 can be run directly in MATLAB.  
2. Experiment 2 is recommended to be run in PyCharm with the Python 3.7 (tensorflow) interpreter (included in the Codes), but it is more advisable to use online platforms like Colab or Kaggle.  
3. Experiment 3 can be run directly in PyCharm by installing the required libraries.

## Experiment 4: Recurrent Neural Network for Speech Denoising
### Content
- **Data Preparation and Preprocessing**: Load the NOIZEUS dataset containing clean and noisy audio samples. Preprocess the audio by extracting STFT-based spectrograms and segmenting them into fixed-size inputs for the GRU-based RNN model.
- **Custom Dataset and DataLoader**: Implement a custom PyTorch Dataset class to handle noisy and clean audio pairs, and use DataLoader for batch processing and shuffling.
- **GRU-based RNN Model**: Construct an 8-layer GRU network with sigmoid activation for predicting an ideal ratio mask (IRM) to denoise speech signals.
- **Training and Validation**: Train the model using MSE loss and the Adam optimizer, with the option to resume training from a pre-trained checkpoint. Validate on a held-out subset.
- **Visualization and Evaluation**: Develop functions to visualize waveforms and spectrograms, and to play audio before and after denoising for qualitative assessment.
- **Hyperparameter Tuning**: Experiment with different numbers of training epochs, learning rates, optimizers (Adam vs. SGD), activation functions (Sigmoid vs. ReLU), and network depths to analyze their impact on denoising performance.
### Results
- The baseline GRU model achieved effective denoising with a final loss of approximately 0.071.
- Increasing the number of training epochs (up to 24) consistently improved denoising performance, though gains diminished after 12 epochs.
- A learning rate of 1e-4 yielded the best results; values too high (e.g., 1e-3) led to training instability.
- The Adam optimizer outperformed SGD significantly in terms of convergence speed and final loss.
- Both Sigmoid and ReLU activations performed similarly well in the GRU network.
- A network depth of 4 to 8 layers provided optimal performance; deeper networks (e.g., 16 layers) showed no further improvement and even degradation.
- Visualization of waveforms and spectrograms confirmed the model’s ability to reduce noise while preserving speech content.

**About Running**:  
This experiment is implemented in PyTorch. It is recommended to run it in a Python environment with PyTorch and torchaudio installed. GPU acceleration is supported but requires proper configuration of CUDA and related libraries.
