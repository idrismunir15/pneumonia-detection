import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pydicom
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

# Set paths
DATA_DIR = 'data/'
TRAIN_DICOM_DIR = os.path.join(DATA_DIR, 'stage_2_train_images')
TEST_DICOM_DIR = os.path.join(DATA_DIR, 'stage_2_test_images')
LABELS_CSV = os.path.join(DATA_DIR, 'stage_2_train_labels.csv')
TEMP_DIR = os.path.join(DATA_DIR, 'temp_images')
MODEL_PATH = 'models/pneumonia_model.h5'
FIGURES_DIR = 'figures/'

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller for simplicity

# Create temporary directories for preprocessed images
for subdir in ['train/normal', 'train/pneumonia', 'test/normal', 'test/pneumonia']:
    os.makedirs(os.path.join(TEMP_DIR, subdir), exist_ok=True)

# Preprocess DICOM to PNG
def preprocess_dicom(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    if len(img.shape) == 2:  # Convert grayscale to RGB
        img = np.stack([img] * 3, axis=-1)
    img = tf.image.resize(img, IMG_SIZE).numpy()
    img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
    print(img)
    plt.imsave(output_path, img)

# Prepare dataset
labels_df = pd.read_csv(LABELS_CSV).drop_duplicates(subset=['patientId'])
train_df = labels_df.sample(frac=0.8, random_state=42)
test_df = labels_df.drop(train_df.index)#.sample(frac=0.2, random_state=42)  # Small test set

# Convert DICOMs to PNG for train and test
for df, split in [(test_df, 'test'), (train_df, 'train')]:
    print(split)
    for _, row in df.iterrows():
        patient_id = row['patientId']
        label = 'pneumonia' if row['Target'] == 1 else 'normal'
        dicom_path = os.path.join(DATA_DIR, f'stage_2_train_images', f'{patient_id}.dcm')
        print(dicom_path)
        output_path = os.path.join(TEMP_DIR, split, label, f'{patient_id}.png')
        if os.path.exists(dicom_path):
            preprocess_dicom(dicom_path, output_path)
            print("saved")
        else:
            print(f"{output_path} failed")
            

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(TEMP_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    os.path.join(TEMP_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)
test_generator = test_datagen.flow_from_directory(
    os.path.join(TEMP_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=10,  # Fewer epochs for simplicity
    validation_data=val_generator
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
y_pred = model.predict(test_generator)
y_pred_binary = (y_pred > 0.5).astype(int)
y_true = test_generator.classes
cm = confusion_matrix(y_true, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
plt.close()

# Save model
model.save(MODEL_PATH)

# Clean up temporary directory
shutil.rmtree(TEMP_DIR)