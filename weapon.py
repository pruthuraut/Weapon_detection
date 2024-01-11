import cv2
import numpy as np
from keras.models import load_model

# Load the Keras model
model = load_model("keras_model.h5")

# Load the labels
class_names = open("labels.txt").read().splitlines()

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Resize frame to match model's expected input shape
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)

    # Normalize the frame
    preprocessed_frame = preprocessed_frame.astype('float32') / 255.0

    # Make predictions
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    # Display prediction
    cv2.putText(frame, predicted_class, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
camera.release()
cv2.destroyAllWindows()
