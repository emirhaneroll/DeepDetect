import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 160

cnn_model = load_model("models/cnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)
    arr = preprocess_input(arr)

    model_input = np.expand_dims(arr, axis=0)
    return model_input

def test_folder(folder, true_label, count=30, fake_index=0):
    files = random.sample(os.listdir(folder), min(count, len(os.listdir(folder))))

    cnn_correct = 0
    lstm_correct = 0
    avg_correct = 0

    for file in files:
        path = os.path.join(folder, file)

        try:
            model_input = prepare_image(path)

            cnn_pred = cnn_model.predict(model_input, verbose=0)[0]
            lstm_pred = lstm_model.predict(model_input, verbose=0)[0]

            cnn_fake_score = cnn_pred[fake_index]
            lstm_fake_score = lstm_pred[fake_index]
            avg_score = (cnn_fake_score + lstm_fake_score) / 2

            cnn_label = 1 if cnn_fake_score > 0.5 else 0
            lstm_label = 1 if lstm_fake_score > 0.5 else 0
            avg_label = 1 if avg_score > 0.5 else 0

            if cnn_label == true_label:
                cnn_correct += 1
            if lstm_label == true_label:
                lstm_correct += 1
            if avg_label == true_label:
                avg_correct += 1

        except Exception as e:
            print("Hata:", file, e)

    print("CNN başarı:", round(cnn_correct / len(files) * 100, 2), "%")
    print("LSTM başarı:", round(lstm_correct / len(files) * 100, 2), "%")
    print("Ortalama başarı:", round(avg_correct / len(files) * 100, 2), "%")

print("\nFAKE_INDEX = 0 TEST")
print("\nREAL TEST")
test_folder("dataset/real", 0, fake_index=0)

print("\nFAKE TEST")
test_folder("dataset/fake", 1, fake_index=0)

print("\nFAKE_INDEX = 1 TEST")
print("\nREAL TEST")
test_folder("dataset/real", 0, fake_index=1)

print("\nFAKE TEST")
test_folder("dataset/fake", 1, fake_index=1)