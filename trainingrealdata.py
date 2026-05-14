"""
TRAIN ON REAL ASL SIGNS
Uses actual ASL signs (DRINK, BOOK, CHAIR, etc) recorded by user
This is the CORRECT approach for real sign language!
"""

import numpy as np
import os
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pickle

def load_real_asl_signs(data_dir='real_asl_data'):
    """Load actual ASL sign videos"""
    
    X = []
    y = []
    sign_to_id = {}
    
    print("\n" + "="*70)
    print("LOADING REAL ASL SIGNS")
    print("="*70)
    print(f"\nLoading from: {data_dir}\n")
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: {data_dir} not found!")
        print("Please run: python collect_real_asl_signs.py")
        return None, None, None
    
    # Get all signs
    sign_files = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            # Extract sign name from filename (e.g., "DRINK_video_0.npy" → "DRINK")
            sign_name = filename.rsplit('_video_', 1)[0]
            if sign_name not in sign_files:
                sign_files[sign_name] = []
            sign_files[sign_name].append(filename)
    
    sign_id = 0
    for sign_name in sorted(sign_files.keys()):
        sign_to_id[sign_id] = sign_name
        
        print(f"  Loading {sign_name}...", end="")
        
        video_count = 0
        for video_file in sorted(sign_files[sign_name]):
            # Load video data
            video_data = np.load(os.path.join(data_dir, video_file))
            
            # Normalize to 30 frames
            if len(video_data) > 30:
                # Take every nth frame to get 30
                step = len(video_data) // 30
                video_data = video_data[::step][:30]
            elif len(video_data) < 30:
                # Pad with repeat of last frame
                padding = np.tile(video_data[-1], (30 - len(video_data), 1))
                video_data = np.vstack([video_data, padding])
            
            X.append(video_data)
            y.append(sign_id)
            video_count += 1
        
        print(f" {video_count} videos")
        sign_id += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Loaded {len(X)} videos from {len(sign_to_id)} signs")
    print(f"  Shape: {X.shape}")
    
    return X, y, sign_to_id


def build_model(num_classes):
    """Build LSTM model for ASL signs"""
    
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(30, 63)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, y, num_classes):
    """Train on real ASL data"""
    
    print("\n" + "="*70)
    print("TRAINING ON REAL ASL SIGNS")
    print("="*70)
    
    # Normalize
    print("\nNormalizing data...")
    X_flat = X.reshape(-1, 63)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(len(X), 30, 63)
    
    # Build model
    print("Building model...")
    model = build_model(num_classes)
    print(model.summary())
    
    # Train
    print(f"\nTraining on {len(X)} videos, {num_classes} signs...")
    y_cat = to_categorical(y, num_classes)
    
    history = model.fit(
        X, y_cat,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Save
    print("\nSaving model...")
    model.save('real_asl_model.h5')
    pickle.dump(scaler, open('real_asl_scaler.pkl', 'wb'))
    print("✓ Model saved: real_asl_model.h5")
    
    return model, scaler, history


def evaluate_model(model, X, y, sign_to_id):
    """Show accuracy per sign"""
    
    print("\n" + "="*70)
    print("ACCURACY BY SIGN")
    print("="*70)
    
    predictions = model.predict(X, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    for sign_id in sorted(sign_to_id.keys()):
        mask = y == sign_id
        if np.sum(mask) > 0:
            accuracy = np.mean(pred_labels[mask] == y[mask])
            sign_name = sign_to_id[sign_id]
            
            # Show accuracy bar
            bar_length = 30
            filled = int(bar_length * accuracy)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"{sign_name:15} {bar} {accuracy*100:6.1f}%")


def test_on_webcam(model, scaler, sign_to_id):
    """Real-time test with your ASL signs"""
    
    print("\n" + "="*70)
    print("REAL-TIME TEST - YOUR ASL SIGNS")
    print("="*70)
    print("\nTest your ASL signs on webcam!")
    print("Press 'q' to quit\n")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    frame_sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Extract landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            hand_data = np.array([
                [lm.x, lm.y, lm.z] for lm in landmarks.landmark
            ]).flatten()
            
            frame_sequence.append(hand_data)
            if len(frame_sequence) > 30:
                frame_sequence.pop(0)
            
            # Predict if we have enough frames
            if len(frame_sequence) >= 30:
                # Normalize
                seq = np.array(frame_sequence)
                seq = scaler.transform(seq)
                seq = seq.reshape(1, 30, 63)
                
                # Predict
                pred = model.predict(seq, verbose=0)
                sign_id = np.argmax(pred)
                confidence = pred[0][sign_id]
                
                sign_name = sign_to_id[sign_id]
                
                # Display
                cv2.putText(frame, f"{sign_name} ({confidence:.0%})",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           (0, 255, 0), 2)
                
                # Show all predictions
                sorted_indices = np.argsort(pred[0])[::-1][:3]
                y_offset = 100
                for idx in sorted_indices:
                    sign = sign_to_id[idx]
                    conf = pred[0][idx]
                    cv2.putText(frame, f"{sign}: {conf:.0%}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
                    y_offset += 30
        
        cv2.imshow("Real ASL Sign Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Test complete!")


def main():
    print("\n" + "="*70)
    print("TRAIN ON YOUR REAL ASL SIGNS")
    print("="*70)
    
    # Load data
    X, y, sign_to_id = load_real_asl_signs('real_asl_data')
    
    if X is None:
        return
    
    num_classes = len(sign_to_id)
    
    # Train
    model, scaler, history = train_model(X, y, num_classes)
    
    # Evaluate
    evaluate_model(model, X, y, sign_to_id)
    
    # Save signs for later use
    pickle.dump(sign_to_id, open('real_asl_signs.pkl', 'wb'))
    
    # Test
    print("\nReady to test? Press ENTER...")
    input()
    test_on_webcam(model, scaler, sign_to_id)
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)
    print(f"\nYour model is trained on {num_classes} real ASL signs!")
    print(f"Files saved:")
    print(f"  - real_asl_model.h5 (trained model)")
    print(f"  - real_asl_scaler.pkl (normalization)")
    print(f"  - real_asl_signs.pkl (sign names)")


if __name__ == "__main__":
    main()