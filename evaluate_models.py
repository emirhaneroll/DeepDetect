import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

cnn_model = load_model("models/cnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_image(path):
    img = prepare_image(path)

    cnn_pred = cnn_model.predict(img, verbose=0)[0]
    lstm_pred = lstm_model.predict(img, verbose=0)[0]

    # Softmax çıktısı: [fake, real]
    cnn_real_score = cnn_pred[0] * 100
    cnn_fake_score = cnn_pred[1] * 100

    lstm_real_score = lstm_pred[0] * 100 
    lstm_fake_score = lstm_pred[1] * 100

    final_score = (cnn_fake_score * 0.7) + (lstm_fake_score * 0.3)

    if final_score > 20:
        return "fake", final_score
    elif final_score > 5:
        return "suspicious", final_score
    else:
        return "real", final_score

def evaluate_folder(folder, expected_label, limit=100):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ][:limit]

    correct = 0
    wrong = 0
    suspicious = 0

    for file in files:
        path = os.path.join(folder, file)

        try:
            prediction, score = predict_image(path)

            if prediction == "suspicious":
                suspicious += 1
            elif prediction == expected_label:
                correct += 1
            else:
                wrong += 1

        except Exception as e:
            print("Hata:", file, e)

    print(f"\n{expected_label.upper()} klasörü sonucu:")
    print("Doğru:", correct)
    print("Yanlış:", wrong)
    print("Şüpheli:", suspicious)
    print("Toplam:", len(files))

evaluate_folder("dataset/real", "real", limit=100)
evaluate_folder("dataset/fake", "fake", limit=100)