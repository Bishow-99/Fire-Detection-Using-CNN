# Fire-Detection-Using-CNN
Fire detection using Convolutional Neural Networks (CNNs) is a popular and effective approach for automatically identifying and detecting fires in images or videos. CNNs are well-suited for this task as they can automatically learn relevant features from the visual data, enabling them to distinguish between fire and non-fire regions.


# Dataset
The dataset is downloaded from Kaggle containing a total of 1832 images, which I subsequently divided into separate training and validation sets. Specifically, I allocated 70% of the dataset, comprising approximately 1282 images, for training purposes. The remaining 30% of the dataset, amounting to around 550 images, was set aside for validation. Furthermore, I have established a distinct testing set, encompassing 68 images. This testing set will be used to assess the performance of the trained model.




# Approach
I have trained a CNN model using the help of TensFlow and PyTorch Framework, however, I have created the same CNN architecture in both frameworks. Before jumping to other steps let's first understand how CNN works in detail:

<b>Convolutional Layers:</b> These are the core building blocks of a CNN. They apply small filters (also known as kernels) to input images. Each filter looks for specific patterns like edges, corners, or textures. Convolutional layers help the model learn to recognize important visual features.

<b>Pooling Layers:</b> After each convolutional layer, pooling layers are often added. These layers reduce the spatial dimensions of the image, making the network computationally more efficient. Common pooling methods include max pooling, where the largest value in each patch is taken, and average pooling, which computes the average.

<b>Activation Functions:</b> Activation functions like ReLU (Rectified Linear Activation) introduce non-linearity to the network. They help the model capture complex relationships between features in the data.

<b>Flatten Layer:</b> This layer reshapes the multi-dimensional output from the convolutional and pooling layers into a one-dimensional vector. It prepares the data for the fully connected layers.

<b>Fully Connected Layers:</b> Also known as dense layers, these layers are similar to traditional neural networks. They process the flattened features and make final predictions. The last fully connected layer usually outputs the class probabilities or scores.

<b>Output Layer:</b> This layer produces the final prediction. In a classification task, it might use softmax activation to convert scores into probabilities for each class.

<b>Loss Function:</b> This measures how well the predicted output matches the actual target (ground truth). Common loss functions include categorical cross-entropy for classification tasks.

<b>Optimization Algorithm:</b> This updates the network's parameters (weights and biases) based on the calculated loss. Common optimization algorithms include stochastic gradient descent (SGD) and its variants.

<b>Training and Backpropagation:</b> The network is trained using labeled data. During training, it adjusts its parameters to minimize the loss by propagating the error backward through the network layers. This process is known as backpropagation.


# External Image
Despite performing validation and testing on the model, I decided to further assess its performance by downloading images from the internet and conducting additional predictions. Encouragingly, the model demonstrated accurate predictions, indicating that it has not suffered from overfitting.


![Screenshot from 2023-08-06 00-41-09](https://github.com/Bishow-99/Fire-Detection-Using-CNN/assets/80660041/01a866a9-90ce-4a7a-91f6-4647f5e12153)


However, if you want to make a more generalized then you can follow the following steps:

<b>Data Augmentation:</b> Data augmentation is a technique used to increase the diversity of the training data by applying various transformations like rotations, flips, and translations to the existing samples. It helps the model generalize better by exposing it to different variations of the same data, making it more robust to different angles and viewpoints.

<b>Reducing Dense Layer and Nodes:</b> When working with a small dataset, it's essential to avoid adding a large number of layers or neurons in each layer, as it can lead to overfitting. Reducing the size of dense layers can prevent the model from memorizing the training data and enable it to generalize better. Smaller models also train faster and use fewer computational resources.

<b>Activation Functions:</b> Choosing appropriate activation functions is crucial for the performance and stability of the neural network. ReLU (Rectified Linear Unit) activation is commonly used as it helps with the vanishing gradient problem, but it can suffer from the "dying ReLU" problem. Variants like Leaky ReLU, Parametric ReLU, ELU (Exponential Linear Unit), and SELU (Scaled Exponential Linear Unit) are better options as they address some of the limitations of ReLU and promote better learning and generalization.

<b>Learning Rate:</b> Tuning the learning rate is important to find the right balance between fast convergence and stability during training. A well-chosen learning rate can prevent the vanishing gradient problem and lead to more efficient and effective training.

<b>Use Dropout Layer:</b> Dropout is a regularization technique where, during training, randomly selected neurons are temporarily "dropped out" by setting their outputs to zero. This helps prevent the model from relying too much on specific neurons and encourages more robust and generalized representations.

<b>Regularization (L1 and L2):</b> Regularization is used to reduce overfitting by adding penalty terms to the loss function. L1 regularization adds the absolute values of the weights to the loss, promoting sparsity, while L2 regularization adds the squared values of the weights, preventing large weight values. Both techniques encourage the model to learn more relevant features and reduce the risk of overfitting.

<b>Other Optimization Techniques:</b> Additional techniques like early stopping, adjusting batch size, and using adaptive learning rate methods (e.g., Adam, RMSprop) can further enhance training efficiency and generalization.

