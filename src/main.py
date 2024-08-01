# Deep Neural Network Implementation for Ransomware Family Classification
# Dataset Utilized: Ransomware RISS Dataset https://rissgroup.org/ransomware-dataset/
# Daniele Sgandurra, Luis Muñoz-González, Rabih Mohsen, Emil C. Lupu. “Automated Analysis
# of Ransomware: Benefits, Limitations, and use for Detection.” In arXiv preprints arXiv:1609.03020, 2016.
# Please see the above link to view the dataset, it is too large to upload on GitHub.

# Written in Spyder IDE using Python 3.18
# Created by: Leo Martinez III in Summer 2024

# Necessary Imports
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model

#%%

# Read the CSV file containing the ransomware data
dataset = pd.read_csv('RansomwareData.csv', header=None)

#%%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the data into features and target
X = dataset.iloc[:, 3:].values # exclude index and label values
# y = dataset[1] # binary class
y = dataset[2] # multiclass

# Split the data into training and testing sets (60/40 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


#%%

""" Deep Neural Network for 12-class Classification (Softmax Regression)""" 

# Import necessary libraries
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Number of features in the input data (30967 total binary features)
n_inputs = 30967

# Define the input layer
visible = Input(shape=(n_inputs,))

# Hidden Layer 1
e = Dense(256, activation='relu')(visible)  # 256 neurons with ReLU activation

# Hidden layer 2
e = Dense(128, activation='relu')(e) # 128 neurons with ReLU activation

# Hidden Layer 3
e = Dense(64, activation='relu')(e) # 64 neurons with ReLU activation

# Hidden Layer 4
e = Dense(32, activation='relu')(e) # 32 neurons with ReLU activation

# Output Layer
output = Dense(12, activation='softmax')(e) # 12 neurons (for 12 classes)

# Define the model
model = Model(inputs=visible, outputs=output)

# Compile the model with sparse categorical crossentropy (lr = 0.0005)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping with a patience of 6 steps
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Fit the model with batch size of 16 and 10 epochs
history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2, validation_split=0.15, callbacks=[early_stopping])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as a 400 dpi .png file
plt.savefig('training_validation_accuracy_plot.png', dpi=400)

plt.show()

# Save the model (can be reused)
model.save('dnn_ransomware.keras')

#--------------------------------------------------------------------------------------------------------------------
# SoftMax Regression Multiclass Classification (12 class)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report and confusion matrix on the test set
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes), "\n")

# Calculate MCC
mcc_score = matthews_corrcoef(y_test, y_pred_classes)
print("MCC Score:", mcc_score, "\n")

#--------------------------------------------------------------------------------------------------------------------

# Generate a confusion matrix (normalized) to better visualize results
def plot_confusion_matrix_heatmap_with_values(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    sorted_classes = sorted(list(classes))  # Convert set to list and sort it
    plt.xticks(tick_marks, sorted_classes, rotation=45)
    plt.yticks(tick_marks, sorted_classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

class_numbers = {0,1,2,3,4,5,6,7,8,9,10,11}

# Original Confusion Matrix
cm_original = confusion_matrix(y_test, y_pred_classes)

# Normalize the confusion matrix
cm_normalized = cm_original.astype('float') / cm_original.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix_heatmap_with_values(cm_normalized, class_numbers, 'Normalized Confusion Matrix')

# Call plt.tight_layout() before saving
plt.tight_layout()

plt.savefig('Heatmap_400dpi.png', dpi=400)  # Saving with 400 dpi
