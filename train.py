from sklearn.datasets import load_files       
#from keras.utils import to_categorical
import numpy as np
from glob import glob
import os
import numpy as np
from glob import glob
from sklearn.datasets import load_files
#from keras.utils import to_categorical
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D


# Set the number of categories
tar = 8
path = './dataset/'


# Create function to load datasets (train and test)
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = to_categorical(np.array(data['target']), tar)
    return files, targets

# Load train and test datasets
train_files, train_targets = load_dataset(path + 'train')  # Path to the train folder
test_files, test_targets = load_dataset(path + 'test')    # Path to the test folder

# Get the deficiency classes (i.e., subfolders inside 'train' and 'test')
#burn_classes = [item.split('/')[-2] for item in sorted(glob(path + "*/"))]
# Get the deficiency classes (i.e., subfolders inside 'train' and 'test')
import os

# Get the deficiency classes (i.e., subfolders inside 'train' and 'test')
train_folders = [f for f in os.listdir(path + 'train') if os.path.isdir(os.path.join(path + 'train', f))]
test_folders = [f for f in os.listdir(path + 'test') if os.path.isdir(os.path.join(path + 'test', f))]

burn_classes = sorted(set(train_folders + test_folders))

# Print statistics about the dataset
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.' % len(test_files))




# Assert no .DS_Store files in train files (if using macOS)
for file in train_files: assert('.DS_Store' not in file)

# Convert image paths to tensors
def path_to_tensor(img_path, width=224, height=224):
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Pre-process the data for Keras (normalize)
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

# Model setup
img_width, img_height = 224, 224
batch_size = 32
epoch = 100
# Model architecture
model = Sequential()


# First Convolutional Layer
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


### Flattening the 3D output to 1D
model.add(Flatten())
##
### Dense Layer with 256 neurons
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Output Layer with softmax activation for classification
model.add(Dense(tar, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=0.0004),
              metrics=['accuracy'])

# Model summary
model.summary()
print(model.summary)
# Train the model with validation split
history = model.fit(train_tensors, train_targets, validation_split=0.3, epochs=epoch, batch_size=10)

# Save the model
model.save('trained_model_DNN1.h5')

# Function to plot training history
def show_history_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Show the accuracy graph
show_history_graph(history)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_tensors, test_targets, verbose=0)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions from the model
test_predictions = model.predict(test_tensors)
test_predictions_classes = np.argmax(test_predictions, axis=1)

# Get the true labels
test_true_classes = np.argmax(test_targets, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(test_true_classes, test_predictions_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=burn_classes, yticklabels=burn_classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# Generate classification report (precision, recall, f1-score, support)
report = classification_report(test_true_classes, test_predictions_classes, target_names=burn_classes)

# Print the classification report
print("Classification Report:\n", report)
