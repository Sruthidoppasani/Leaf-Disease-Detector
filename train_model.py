import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    'Dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
# Get correct class label order
label_map = train_data.class_indices
class_labels = [None] * len(label_map)
for label, idx in label_map.items():
    class_labels[idx] = label

print("✅ Class label order:")
print(class_labels)
np.save("class_labels.npy", class_labels)

val_data = datagen.flow_from_directory(
    'Dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("leaf_model.h5", save_best_only=True)

# Phase 1: Train with frozen base
model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[early_stop])

# Phase 2: Fine-tune top layers
#base_model.trainable = True
#for layer in base_model.layers[:100]:
#   layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[early_stop, checkpoint])

# Evaluate
loss, acc = model.evaluate(val_data)
print(f"✅ Final Validation Accuracy: {acc:.2f}")

print("✅ Training complete. Model saved as 'leaf_model.h5'")


# Save the trained model
model.save("leaf_model.h5")

print("✅ Training complete. Model saved as 'leaf_model.h5'")
