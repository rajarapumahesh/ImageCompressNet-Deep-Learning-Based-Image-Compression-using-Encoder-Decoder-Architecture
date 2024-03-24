# ImageCompressNet-Deep-Learning-Based-Image-Compression-using-Encoder-Decoder-Architecture
The provided code implements an autoencoder for compressing and reconstructing chest CT scan images. Let's break down the project in depth:

### 1. Data Preparation:
- **Dataset**: The dataset consists of chest CT scan images, specifically focusing on COVID-19-related cases. The dataset directory is specified by `folder_path`.
- **Data Loading**: The `load_images` function loads the grayscale images from the specified directory, resizes them to a predefined size (256x256), and normalizes pixel values to the range [0, 1].
- **Data Splitting**: The dataset is split into training and testing sets using `train_test_split` from scikit-learn. The split ratio is set to 80% training and 20% testing.

### 2. Autoencoder Architecture:
- **Model Definition**: The autoencoder architecture is defined using the Keras Functional API.
- **Encoder**: The encoder part consists of convolutional layers followed by max-pooling layers, gradually reducing the spatial dimensions while increasing the number of channels.
- **Decoder**: The decoder part mirrors the encoder architecture but uses upsampling layers to reconstruct the original spatial dimensions.
- **Model Compilation**: The model is compiled with the Adam optimizer and binary cross-entropy loss, as it's a binary image reconstruction task.

### 3. Model Training:
- **Training**: The model is trained using the training set with a batch size of 16 and for 10 epochs.
- **Data Augmentation** (optional): Data augmentation techniques can be applied to increase the diversity of the training data and improve model generalization.
- **Model Checkpoints** (optional): Model weights are saved during training using checkpoints to retain the best-performing model.
- **Learning Rate Scheduler** (optional): A scheduler adjusts the learning rate during training to improve convergence.

### 4. Evaluation and Visualization:
- **Evaluation**: The trained model is evaluated on the testing set.
- **Visualization**: The original and compressed images from the testing set are visualized side by side for qualitative assessment. Additionally, loss curves can be plotted to assess training and validation performance.

### 5. Project Enhancements (Additional Suggestions):
- **Data Augmentation**: Augmenting the data can help improve the model's robustness and performance.
- **Model Checkpoints**: Saving the best model weights allows for easy retrieval of the top-performing model.
- **Learning Rate Scheduler**: Adjusting the learning rate dynamically can help in faster convergence and better optimization.
- **Visualization Improvements**: Enhanced visualizations provide better insights into model performance and reconstruction quality.

### 6. Potential Applications:
- **Medical Imaging**: This project can be applied in medical imaging tasks, particularly in processing and analyzing chest CT scans for various pulmonary conditions, including COVID-19.
- **Compression**: Autoencoders are commonly used for data compression tasks, where high-dimensional data can be efficiently represented in a lower-dimensional space.
- **Anomaly Detection**: Autoencoders can also be used for anomaly detection by reconstructing input data and comparing it with the original data, identifying any significant differences.

In summary, this project demonstrates the implementation of an autoencoder for compressing and reconstructing chest CT scan images, with potential applications in medical imaging and data compression tasks. Additionally, it showcases techniques such as data augmentation, model checkpoints, and learning rate scheduling for model training and improvement.
