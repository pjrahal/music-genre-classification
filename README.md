🎵 Music Genre Classifier

A PyTorch-based deep learning project for automatic music genre classification using mel-spectrogram features from the FMA dataset.

📁 Project Structure

├── settings.py               # Shared constants and logger

├── setup.py                  # Sets up virtualenv and installs packages

├── main.py                   # Entry point for training/testing

├── getData.py                # Dataset preparation and loading logic

├── train.py                  # Training loop

├── test.py                   # Evaluation loop

├── genreCNN.py               # CNN model definition

├── cachedGenreDataset.py     # Dataset class with feature caching

├── audioPreprocessing.py     # Audio preprocessing script

├── features/                 # Extracted mel-spectrograms and metadata

├── models/                   # Saved model checkpoints

├── fma_small/                # Raw audio files

├── fma_metadata/             # Track metadata

⚙️ Setup

# 1. Clone the repository
$ git clone [https://github.com/your-repo/music-genre-classifier.git](https://github.com/pjrahal/music-genre-classification)

$ cd music-genre-classifier

# 2. Run setup script (creates venv, installs deps, writes requirements.txt)
$ python setup.py

# 3. Activate virtual environment
$ source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows

🎧 Preprocessing Audio

This step will:

Download the FMA dataset and metadata (if not already downloaded)

Compute mel-spectrogram features and deltas

Normalize using global stats

Save as .npy files and generate metadata CSV

$ python audioPreprocessing.py

🚀 Training & Testing

Run the full training pipeline:

$ python main.py

This will:

Load preprocessed features

Train a CNN on the training set

Evaluate and save the best model

Print test set accuracy and confusion matrix

🧠 Model

The CNN model (GenreCNN) consists of:

3 convolutional layers with grouped conv, batch norm, ReLU

Adaptive average pooling

Dropout and fully connected classification layer

📝 Configurable Parameters

Defined in config.py:

Batch size, learning rate, number of epochs

Dataset paths, model save location

📄 Requirements

Automatically installed by setup.py. If needed:

pip install -r requirements.txt

📈 Output

features/ contains .npy mel features and features_metadata.csv

models/best_model_val.pth stores best model based on validation loss

Built with 💙 using PyTorch and Librosa
