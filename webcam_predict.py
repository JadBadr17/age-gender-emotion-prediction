import cv2
import numpy as np
import joblib

# Load emotion detection model
emotion_model, label_map, scaler, pca = joblib.load('model/emotion_model.pkl')
reverse_label_map = {v: k for k, v in label_map.items()}

# Load age & gender model
age_gender_model = joblib.load('model/age_gender_model.pkl')

# Convert single label into age group and gender
def decode_age_gender(label):
    
    age_group = label // 2
    gender = label % 2
    age_ranges = ["0-10", "11-20", "21-30", "31-40", "41-50", "51+"]
    gender_str = "Male" if gender == 0 else "Female"
    return age_ranges[age_group], gender_str

def predict_all():
    cap = cv2.VideoCapture(1)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            face_emotion = cv2.resize(face, (128, 128)).flatten().reshape(1, -1)
            face_scaled = scaler.transform(face_emotion)
            face_pca = pca.transform(face_scaled)
            emotion_pred = emotion_model.predict(face_pca)[0]
            emotion = reverse_label_map[emotion_pred]

            face_ag = cv2.resize(face, (129,128)).flatten().reshape(1, -1)
            ag_label = age_gender_model.predict(face_ag)[0]
            age_group, gender = decode_age_gender(ag_label)

            label_text = f"{gender}, {age_group}, {emotion}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Emotion, Age & Gender Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_all()
