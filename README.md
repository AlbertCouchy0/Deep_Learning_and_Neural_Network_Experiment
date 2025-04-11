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

## Experiment 2: Experimental Environment Configuration and Convolutional Neural Network
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
- The spam classification model showed high accuracy, with certain词汇 having high weights for spam identification.

  ## Running Steps
