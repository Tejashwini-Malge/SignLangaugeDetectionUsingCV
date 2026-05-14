"""
REAL ASL SIGN COLLECTION
Record actual ASL signs like DRINK, BOOK, CHAIR, etc.
Not simple gestures!

This is the RIGHT approach for real sign language recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

os.makedirs('real_asl_data', exist_ok=True)

# Common ASL signs
COMMON_ASL_SIGNS = {
    1: "DRINK",
    2: "BOOK",
    3: "CHAIR",
    4: "FOOD",
    5: "WATER",
    6: "HELP",
    7: "HELLO",
    8: "GOODBYE",
    9: "THANK_YOU",
    10: "LOVE",
    11: "FRIEND",
    12: "FAMILY",
    13: "HAPPY",
    14: "SAD",
    15: "BEAUTIFUL",
    16: "PERSON",
    17: "HOUSE",
    18: "SCHOOL",
    19: "WORK",
    20: "PLAY",
}

def show_asl_signs():
    """Show available ASL signs"""
    print("\n" + "="*70)
    print("AVAILABLE ASL SIGNS TO RECORD")
    print("="*70 + "\n")
    
    for num, sign in COMMON_ASL_SIGNS.items():
        print(f"  {num:2d}. {sign}")
    
    print("\n" + "="*70)


def get_user_sign_selection():
    """Ask user which signs to record"""
    
    show_asl_signs()
    
    print("\nWhich signs do you want to record?")
    print("Options:")
    print("  1. All (1-20)")
    print("  2. First 5 (DRINK, BOOK, CHAIR, FOOD, WATER)")
    print("  3. Custom (you choose)")
    
    choice = input("\nChoose (1/2/3): ").strip()
    
    if choice == "1":
        selected = list(range(1, 21))
    elif choice == "2":
        selected = list(range(1, 6))
    elif choice == "3":
        print("\nEnter sign numbers separated by commas (e.g., 1,3,5,7):")
        user_input = input("Signs: ").strip()
        try:
            selected = [int(x.strip()) for x in user_input.split(",")]
            selected = [x for x in selected if 1 <= x <= 20]
        except:
            print("Invalid input. Using first 5 signs.")
            selected = list(range(1, 6))
    else:
        print("Invalid choice. Using first 5 signs.")
        selected = list(range(1, 6))
    
    return {num: COMMON_ASL_SIGNS[num] for num in selected}


def record_asl_sign(sign_name, video_idx, total_videos):
    """
    Record ONE video of an ASL sign
    
    Args:
        sign_name: str (e.g., "DRINK")
        video_idx: int (video number for this sign)
        total_videos: int (total videos to record for this sign)
    """
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not working!")
        return False
    
    print(f"\n{'='*70}")
    print(f"Recording: {sign_name}")
    print(f"Video: {video_idx + 1}/{total_videos}")
    print(f"{'='*70}")
    
    print(f"""
📝 How to sign "{sign_name}":
  1. Look up the sign if you don't know it
  2. Position yourself in front of camera
  3. Show the sign clearly
  4. Press SPACE when ready to start
  5. Hold the sign for 3-5 seconds
  6. Press SPACE to stop and save
  
⚠️  Tips:
  - Make sure your WHOLE hand is visible
  - Good lighting is important
  - Don't rush - hold the sign clearly
  - Can redo if you mess up (press 'q' and try again)
    """)
    
    input("Press ENTER when you're ready...")
    
    frames_data = []
    is_recording = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Extract hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Status display
        if is_recording:
            status = f"🔴 RECORDING ({len(frames_data)}/90 frames)"
            color = (0, 0, 255)  # Red
        else:
            status = "⏸️  Press SPACE to record"
            color = (0, 255, 0)  # Green
        
        # Draw text
        cv2.putText(frame, status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Sign: {sign_name}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Save landmarks if recording
            if is_recording:
                hand_data = np.array([
                    [lm.x, lm.y, lm.z] for lm in landmarks.landmark
                ]).flatten()
                frames_data.append(hand_data)
                
                # Draw skeleton
                for i, lm in enumerate(landmarks.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            
            cv2.putText(frame, "✓ Hand detected", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "⚠️  No hand detected", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow(f"Recording: {sign_name}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Toggle recording
            if not is_recording:
                print(f"  ▶️  Started recording...")
                is_recording = True
                frames_data = []
            else:
                print(f"  ⏹️  Stopped recording!")
                is_recording = False
                break
        
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print(f"  ❌ Skipped")
            return False
        
        # Auto-save if we have enough frames
        if is_recording and len(frames_data) >= 90:
            print(f"  ✓ Got enough frames! Saving...")
            is_recording = False
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save data
    if len(frames_data) >= 30:  # At least 1 second
        save_path = f"real_asl_data/{sign_name}_video_{video_idx:02d}.npy"
        np.save(save_path, np.array(frames_data))
        print(f"  ✅ Saved: {save_path} ({len(frames_data)} frames)")
        return True
    else:
        print(f"  ⚠️  Not enough frames ({len(frames_data)}/30). Try again.")
        return False


def collect_asl_signs():
    """Collect multiple ASL signs"""
    
    print("\n" + "="*70)
    print("REAL ASL SIGN COLLECTION")
    print("="*70)
    
    # Get signs to record
    signs_to_record = get_user_sign_selection()
    
    videos_per_sign = int(input(f"\nHow many videos per sign? (5-30): ") or "10")
    videos_per_sign = max(5, min(30, videos_per_sign))
    
    total_signs = len(signs_to_record)
    total_videos = total_signs * videos_per_sign
    
    print(f"\n{'='*70}")
    print(f"COLLECTION PLAN")
    print(f"{'='*70}")
    print(f"Signs: {total_signs}")
    print(f"Videos per sign: {videos_per_sign}")
    print(f"Total videos: {total_videos}")
    print(f"Estimated time: {total_videos * 0.5 / 60:.0f}-{total_videos * 1 / 60:.0f} minutes")
    print(f"{'='*70}\n")
    
    input("Press ENTER to start recording...\n")
    
    total_collected = 0
    
    for sign_idx, (sign_num, sign_name) in enumerate(signs_to_record.items()):
        successful = 0
        attempts = 0
        max_attempts = videos_per_sign + 5  # Allow some failures
        
        while successful < videos_per_sign and attempts < max_attempts:
            success = record_asl_sign(sign_name, successful, videos_per_sign)
            attempts += 1
            
            if success:
                successful += 1
                total_collected += 1
            
            if successful < videos_per_sign and attempts < max_attempts:
                print("\n⏳ Take a break! 10 seconds before next video...\n")
                import time
                for i in range(1, 0, -1):
                    print(f"  Starting in {i}...")
                    time.sleep(1)
        
        progress = (sign_idx + 1) / total_signs * 100
        print(f"\nProgress: {progress:.0f}% ({sign_idx + 1}/{total_signs} signs)")
        
        if successful == videos_per_sign:
            print(f"✅ Completed {sign_name}!")
        else:
            print(f"⚠️  Got {successful}/{videos_per_sign} for {sign_name}")
    
    print(f"\n{'='*70}")
    print(f"✅ COLLECTION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total videos collected: {total_collected}")
    print(f"Location: real_asl_data/")
    print(f"\nNext step: python train_on_real_asl_data.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    collect_asl_signs()