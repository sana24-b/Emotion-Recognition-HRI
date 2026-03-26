import cv2
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

# =========================
# MODEL
# =========================
def build_model():
    inputs = Input(shape=(48, 48, 1))

    x = Conv2D(64, (5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(7, activation='softmax')(x)

    model = Model(inputs, outputs)
    #model.load_weights("my_emotion_model_weights.h5")
    # Pre-trained model weights not included due to size limitations
    # Uncomment the line below and provide the .h5 file to run the model
    return model

model = build_model()

# =========================
# EMOTIONS
# =========================
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def map_emotion(e):
    if e in ['Happy', 'Surprise']:
        return 'Happy'
    elif e in ['Angry', 'Disgust']:
        return 'Angry'
    elif e == 'Sad':
        return 'Sad'
    elif e == 'Fear':
        return 'Fear'
    else:
        return 'Neutral'

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# STABILITY + LATENCY
# =========================
prev = ""
count = 0
cooldown = 0

latencies = []   # for average calculation

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48)) / 255.0
        face = np.reshape(face, (1,48,48,1))

        # =========================
        # LATENCY START
        # =========================
        start = time.time()

        pred = model.predict(face, verbose=0)

        end = time.time()
        latency = end - start

        # store latency
        latencies.append(latency)

        # print per-frame latency
        print("Latency:", round(latency, 4), "sec")

        # print average every 30 frames
        if len(latencies) == 30:
            avg = sum(latencies) / len(latencies)
            print("🔥 Avg Latency:", round(avg, 4), "sec\n")
            latencies = []

        # =========================
        # EMOTION
        # =========================
        emotion = labels[np.argmax(pred)]
        mapped = map_emotion(emotion)

        if mapped == prev:
            count += 1
        else:
            count = 0

        prev = mapped

        # =========================
        # WRITE TO FILE
        # =========================
        if count >= 15 and cooldown == 0:
            with open("emotion.txt", "w") as f:
                f.write(mapped)

            print("Stable:", mapped)
            cooldown = 50

        if cooldown > 0:
            cooldown -= 1

        # =========================
        # DISPLAY
        # =========================
        cv2.putText(frame, mapped, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Emotion", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
