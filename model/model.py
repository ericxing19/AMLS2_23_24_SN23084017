import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Multiply, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB4
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# show the training process
class PrintEpochProgress(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        self.epoch_progress = tqdm(total=self.total_epochs, desc='Epochs', position=0)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.update(1)
        self.epoch_progress.set_postfix(loss=f"{logs['loss']:.4f}", accuracy=f"{logs['accuracy']:.4f}", val_loss=f"{logs['val_loss']:.4f}", val_accuracy=f"{logs['val_accuracy']:.4f}")

# show the training process
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, steps_per_epoch):
        super(CustomCallback, self).__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.progress_bar = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar = tqdm(total=self.steps_per_epoch, desc=f'Epoch {epoch+1}/{self.total_epochs}')
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()
        print(f"    loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")
        
    def on_train_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        
# Attention mask layer
def attention_module(feature_map, input_channels):
    # Use 1x1 convolution to reduce dimensionality
    attention_features = Conv2D(filters=input_channels, kernel_size=(1, 1), padding='same')(feature_map)
    # Generate attention mask
    attention_mask = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(attention_features)
    # Apply attention mask to feature map through multiplication
    attention_applied = Multiply()([feature_map, attention_mask])
    return attention_applied, attention_mask

# global_weighted_average_pooling layer
def global_weighted_average_pooling(feature_map, attention_mask):
    # Weight feature map
    weighted_feature_map = Multiply()([feature_map, attention_mask])
    # Sum attention mask
    sum_attention = tf.reduce_sum(attention_mask, axis=[1, 2], keepdims=True)
    # Weighted average pooling
    weighted_average_pool = weighted_feature_map / (sum_attention + tf.keras.backend.epsilon())
    # Global average pooling
    gap = tf.reduce_sum(weighted_average_pool, axis=[1, 2])
    return gap

def create_primitive_CNN_model(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(5, activation='softmax')
    ])
    return model

def create_attention_based_CNN_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # First Conv2D layer
    x = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Second Conv2D layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Third Conv2D layer
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Fourth Conv2D layer
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Attention module
    x, attention_mask = attention_module(x, input_channels=128)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(512, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model

def create_attention_based_pre_trained_model(input_shape):
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Pre-trained model
    pre_model = EfficientNetB4(include_top=False, input_tensor=inputs)
    x = pre_model.output
    
    # Batch normalization
    x = BatchNormalization()(x)
    
    # First additional convolutional layer
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Second additional convolutional layer
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Attention module
    x, attention_mask = attention_module(x, input_channels=pre_model.output_shape[-1])
    
    # Global weighted average pooling
    x = global_weighted_average_pooling(x, attention_mask)
    
    # Classification head
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(5, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=predictions)
    
    return model

def train_model(model, X_train, y_train, patience = 3, model_save_path = 'best_model.h5', epoch = 30, lr = 0.001, batch_size = 32):
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

    # Define model checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epoch, batch_size=batch_size, 
                        verbose=0, callbacks=[CustomCallback(total_epochs=epoch, steps_per_epoch=len(X_train)//batch_size), early_stopping, checkpoint])
    
    return history

def SVM_pred(model, X_train, X_test, y_train, y_test, C):
    # Get feature map from the model trained before
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

    train_features = feature_extractor.predict(X_train)
    print(train_features.shape)
    test_features = feature_extractor.predict(X_test)

    # Convert multi-class target variable to one-dimensional array
    y_train_flat = np.argmax(y_train, axis=1)
    y_test_flat = np.argmax(y_test, axis=1)

    # train SVM
    svm_model = SVC(kernel='rbf', C = C)
    svm_model.fit(train_features, y_train_flat)

    # evaluate model on test set
    y_pred = svm_model.predict(test_features)
    accuracy = accuracy_score(y_test_flat, y_pred)
    return accuracy

    