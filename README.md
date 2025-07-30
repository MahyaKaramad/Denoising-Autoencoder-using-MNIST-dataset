# 🧠 Denoising Autoencoder on MNIST

This project implements a convolutional autoencoder in TensorFlow/Keras to denoise MNIST digit images. The model is trained to remove noise from input images and reconstruct clean digit images.

## 📌 Project Overview

Autoencoders are neural networks that learn to encode input data into a lower-dimensional representation and decode it back to its original form. This project uses a **denoising autoencoder**, which learns to reconstruct a clean image from a noisy version.

- **Dataset:** MNIST handwritten digits
- **Model:** Convolutional autoencoder with symmetric encoder-decoder
- **Goal:** Learn to remove Gaussian noise from digit images

## 🗃️ Dataset

- **Source:** [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Format:** 28x28 grayscale images of digits (0–9)
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images

## 🏗️ Architecture

- Encoder:
  - Conv2D + ReLU
  - MaxPooling
- Bottleneck:
  - Flatten → Dense (64) → Dense → Reshape
- Decoder:
  - Conv2DTranspose + ReLU
  - Final Conv2D (Sigmoid)

## 📈 Training

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 40
- Batch Size: 128
- Metrics: Accuracy

## 📊 Results

The model successfully removes noise and reconstructs  digits.

| Original | Noisy | Denoised |
|----------|-------|----------|
| ![](examples/original_1.png) | ![](examples/noisy_1.png) | ![](examples/denoised_1.png) |



## 📉 Loss & Accuracy

Training and validation metrics are plotted for performance tracking.

```python
# Sample code to plot history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()
```

## 🧪 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook "DenoisingAutoencoder .ipynb"
   ```

## 🛠️ Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

## 📚 References

- [Medium: Training a Simple Autoencoder on MNIST](https://2020machinelearning.medium.com/training-a-simple-autoencoder-on-the-mnist-dataset-a-hand-on-tutorial-46d8a024604c)
- [Kaggle Notebook](https://www.kaggle.com/code/roblexnana/understanding-auto-encoder-on-mnist-digit-dataset)

## 👩‍💻 Author

Mahya Karamad Kasmaei  
Marin Vratonjic
Aryan Sunilkumar Singh 

M.Eng. Students – McMaster University  
Deep Learning & Computer Vision Enthusiast  

## 📜 License

This project is licensed under the MIT License.
