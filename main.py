from model.model import *
import csv
import matplotlib.pyplot as plt
import os

# The path of the dataset
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
data_path = r"Datasets\10000Dataset224.npy"
path = os.path.abspath(os.path.join(current_directory, data_path))
print(path)
data = np.load(path, allow_pickle=True).item()
# # create DataFrame
df = pd.DataFrame(data)
dataset = df

images_o = np.stack(dataset['image_matrices'].values)
labels_o = dataset['labels'].values
# labels = np.array(labels, dtype=np.int32)

print(images_o.shape)
print(labels_o)

# data preprocessing
images = np.array(images_o).astype('float32') / 255.0  # Normalizaion
labels = tf.keras.utils.to_categorical(labels_o)  # convert label to one-hot code

# split train_test dadtaset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# input
input_shape=(224, 224, 3)

# Create models: chooose 1 from 3 
# primitive_CNN = create_primitive_CNN_model(input_shape)
attention_CNN =  create_attention_based_CNN_model(input_shape)
# attention_pre_trained = create_attention_based_pre_trained_model(input_shape)


# train model: chooose 1 from 3 
# history = train_model(primitive_CNN, X_train, y_train, patience = 3, model_save_path = 'CNN_best_model.h5', epoch = 30, lr = 0.001,batch_size=32)
train_model(attention_CNN, X_train, y_train, patience = 5, model_save_path = 'attention_based_best_model.h5', epoch = 50, lr = 0.002, batch_size=32)
# train_model(primitive_CNN, X_train, y_train, patience = 3, model_save_path = 'attention_pretrain_best_model.h5', epoch = 30, lr = 0.000002, batch_size=32)

# SVM prediction (change model)
# accuracy = SVM_pred(primitive_CNN, X_train, X_test, y_train, y_test, C = 1)
# print("Accuracy:", accuracy)

# evaluate model: chooose 1 from 3 
# loss, accuracy = primitive_CNN.evaluate(X_test, y_test)
loss, accuracy = attention_CNN.evaluate(X_test, y_test)
# loss, accuracy = attention_pre_trained.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)