import os
import sys
import json
import tempfile
import asyncio
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.input_file import InputFile
import numpy as np

try:
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2
    cv2_available = True
except ImportError as e:
    print(f"OpenCV not available: {e}")
    cv2_available = False
    cv2 = None

try:
    import mediapipe as mp
    mp_available = True
except ImportError as e:
    print(f"MediaPipe not available: {e}")
    mp_available = False
    mp = None

async def main(context):
    try:
        endpoint = os.environ.get("APPWRITE_ENDPOINT")
        project = os.environ.get("APPWRITE_FUNCTION_PROJECT_ID")
        api_key = os.environ.get("APPWRITE_API_KEY")
        bucket_id = os.environ.get("APPWRITE_BUCKET_ID")

        event = json.loads(context.req.body)
        file_id = event["fileId"]
        user_id = event.get("userId", "user")
        backswing_id = event.get("backswingId", f"{user_id}_backswing")
        midswing_id = event.get("midswingId", f"{user_id}_midswing")
        followthrough_id = event.get("followthroughId", f"{user_id}_followthrough")

        client = Client()
        client.set_endpoint(endpoint)
        client.set_project(project)
        client.set_key(api_key)
        storage = Storage(client)

        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input.mp4")
        with open(video_path, "wb") as f:
            file = storage.get_file_download(bucket_id, file_id)
            f.write(file)

        key_frames = await run_forehand_analysis(video_path, temp_dir, user_id)

        result_urls = {}
        id_map = {"backswing": backswing_id, "midswing": midswing_id, "followthrough": followthrough_id}
        
        for k, path in key_frames.items():
            if path and os.path.exists(path):
                custom_file_id = id_map.get(k, f"{user_id}_{k}")
                print(f"Uploading {k} with ID: {custom_file_id}")

                uploaded = storage.create_file(
                    bucket_id=bucket_id,
                    file_id=custom_file_id,
                    file=InputFile.from_path(path),
                    permissions=[]
                )
                result_urls[k] = f"{endpoint}/v1/storage/buckets/{bucket_id}/files/{uploaded['$id']}/view?project={project}"

        response_data = {
            "success": True,
            "frames": result_urls
        }
        
        print(f"Function completed successfully")
        print(f"Result URLs: {result_urls}")
        print(f"Response data: {response_data}")
        
        return context.res.json(response_data)
    
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e)
        }
        
        print(f"Function failed with error: {str(e)}")
        print(f"Error response: {error_response}")
        
        return context.res.json(error_response)

async def run_forehand_analysis(video_path, temp_dir, name):
    """Forehand analysis with available libraries"""
    
    if not cv2_available:
        print("OpenCV not available - cannot process video")
        return {}
    
    await asyncio.sleep(0.01)
    
    if mp_available:
        print("Initializing MediaPipe...")
        
        try:
            mp_pose = mp.solutions.pose
            
            pose = mp_pose.Pose(
                static_image_mode=True,  
                model_complexity=0,  
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            mp_drawing = mp.solutions.drawing_utils
            use_mediapipe = True
            print("MediaPipe initialized successfully")
            
        except Exception as e:
            print(f"MediaPipe initialization failed: {e}")
            use_mediapipe = False
            pose = None
            mp_drawing = None
            mp_pose = None
    else:
        use_mediapipe = False
        pose = None
        mp_drawing = None
        mp_pose = None
        print("MediaPipe not available - using fallback detection")
    
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle

    def findBackswingF(landmarks_or_motion, x):
        if use_mediapipe and landmarks_or_motion:
            # MediaPipe version
            landmarks = landmarks_or_motion
            right_hip = landmarks[23].x  # Right hip
            left_hip = landmarks[24].x   # Left hip
            hip_center = (right_hip + left_hip) / 2
            right_wrist_x = landmarks[20].x  # Right wrist
            
            if right_wrist_x < hip_center - 0.08: 
                if x == -1:
                    x = hip_center - right_wrist_x
                    return x, True
                elif (hip_center - right_wrist_x) > x + 0.02:  
                    return hip_center - right_wrist_x, True
                else:
                    return x, False
            else:
                return x, False
        else:
            # Fallback
            motion_data = landmarks_or_motion
            if motion_data is None:
                return x, False
            
            frame_variation = motion_data
            if frame_variation > 0.1: 
                if x == -1:
                    x = frame_variation
                    return x, True
                elif frame_variation > x + 0.05:
                    return frame_variation, True
                else:
                    return x, False
            else:
                return x, False

    def findMidF(landmarks_or_motion):
        if use_mediapipe and landmarks_or_motion:
            # MediaPipe version
            landmarks = landmarks_or_motion
            right_hip = landmarks[23].x  # Right hip
            left_hip = landmarks[24].x   # Left hip
            hip_center = (right_hip + left_hip) / 2
            right_wrist_x = landmarks[20].x  # Right wrist
            
            distance_from_hip = abs(hip_center - right_wrist_x)
            return distance_from_hip
        else:
            # Fallback
            motion_data = landmarks_or_motion
            return motion_data if motion_data is not None else 0.5

    def findFollowF(landmarks_or_motion, x):
        if use_mediapipe and landmarks_or_motion:
            # MediaPipe version
            landmarks = landmarks_or_motion
            right_shoulder_x = landmarks[12].x  # Right shoulder
            right_wrist_x = landmarks[20].x     # Right wrist

            if right_wrist_x > right_shoulder_x + 0.08:  
                dist = right_wrist_x - right_shoulder_x
                if x == -1:
                    x = dist
                    return x, True
                elif dist > x + 0.02: 
                    return dist, True
                else:
                    return x, False
            else:
                return x, False
        else:
            # Fallback
            motion_data = landmarks_or_motion
            if motion_data is None:
                return x, False
            
            if motion_data > 0.15:  
                if x == -1:
                    x = motion_data
                    return x, True
                elif motion_data > x + 0.03:
                    return motion_data, True
                else:
                    return x, False
            else:
                return x, False

    def process_frame_with_pose_or_motion(frame):
        """Process frame to extract pose keypoints or motion data"""
        if use_mediapipe and pose:
            # MediaPipe version
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                return results.pose_landmarks.landmark
            else:
                return None
        else:
            # Fallback
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                motion_score = np.std(gray) / 255.0  # Normalize to 0-1
                
                return motion_score
            except Exception as e:
                print(f"Error in motion detection: {e}")
                return 0.1  # Default motion value

    cap = cv2.VideoCapture(video_path)
    count = 0
    midimages = []
    midd = []
    backframe = -1
    followframe = -1
    xback = -1
    successback = False
    xfollow = -1
    successfollow = False
    middist = -1
    
    swing_phase = "start"  # start -> backswing -> midswing -> followthrough -> end

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        new_height = 300
        new_width = int(new_height * width / height)
        dim = (new_width, new_height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        try:
            detection_data = process_frame_with_pose_or_motion(frame)
            
            image = frame.copy()
            
            if detection_data is not None:
                if use_mediapipe:
                    landmarks = detection_data
                    mp_drawing.draw_landmarks(
                        image, 
                        mp_pose.PoseLandmark.from_landmark_list(landmarks),
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                else:
                    motion_score = detection_data
                    cv2.putText(image, f"Motion: {motion_score:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if swing_phase in ["start", "backswing"]:
                    xback, successback = findBackswingF(detection_data, xback)
                    if successback and swing_phase == "start":
                        swing_phase = "backswing"
                else:
                    successback = False  
                
                middist = findMidF(detection_data)
                
                if swing_phase in ["backswing", "midswing", "followthrough"]:
                    xfollow, successfollow = findFollowF(detection_data, xfollow)
                    if successfollow and swing_phase in ["backswing", "midswing"]:
                        swing_phase = "followthrough"
                else:
                    successfollow = False  
                
                if count % 10 == 0:  
                    if use_mediapipe:
                        landmarks = detection_data
                        right_wrist = landmarks[20]  # Right wrist
                        right_hip = landmarks[23]    # Right hip  
                        left_hip = landmarks[24]     # Left hip
                        print(f"Frame {count}: phase={swing_phase}, wrist_x={right_wrist.x:.3f}, hip_avg={(right_hip.x + left_hip.x)/2:.3f}, backswing={successback}, follow={successfollow}")
                    else:
                        motion_score = detection_data
                        print(f"Frame {count}: phase={swing_phase}, motion={motion_score:.3f}, backswing={successback}, follow={successfollow}")
                
                cv2.putText(image, f"Frame {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detection_text = "POSE DETECTED" if use_mediapipe else "MOTION DETECTED"
                cv2.putText(image, detection_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if successback:
                    cv2.putText(image, "BACKSWING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if successfollow:
                    cv2.putText(image, "FOLLOW-THROUGH", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # No detection
                cv2.putText(image, f"Frame {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                no_detection_text = "NO POSE DETECTED" if use_mediapipe else "NO MOTION DETECTED"
                cv2.putText(image, no_detection_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
        except Exception as e:
            print(f"Error in pose detection: {e}")
            image = frame
            
        if successback:
            cv2.imwrite(os.path.join(temp_dir, f"backswing_{name}.jpg"), image)
            backframe = count
        if successfollow:
            cv2.imwrite(os.path.join(temp_dir, f"followthrough_{name}.jpg"), image)
            followframe = count
            
        midimages.append(image)
        midd.append(middist)
        count += 1
        
        await asyncio.sleep(0.001)
        
        if count >= 100:
            break
            
    cap.release()
    
    if use_mediapipe and pose:
        pose.close()
    
    print(f"Analysis complete. Total frames: {count}")
    print(f"Backswing detected at frame: {backframe if backframe != -1 else 'None'}")
    print(f"Follow-through detected at frame: {followframe if followframe != -1 else 'None'}")
    print(f"Mid-swing images collected: {len(midimages)}")
    
    mind = None
    img = None
    for i in range(len(midimages)):
        if i > backframe and i < followframe:
            if mind is None:
                mind = midd[i]
                img = midimages[i]
            elif midd[i] < mind:
                mind = midd[i]
                img = midimages[i]
    
    if img is not None:
        cv2.imwrite(os.path.join(temp_dir, f"midswing_{name}.jpg"), img)
    
    result = {}
    
    backswing_path = os.path.join(temp_dir, f"backswing_{name}.jpg")
    midswing_path = os.path.join(temp_dir, f"midswing_{name}.jpg")
    followthrough_path = os.path.join(temp_dir, f"followthrough_{name}.jpg")
    
    if os.path.exists(backswing_path):
        result["backswing"] = backswing_path
    else:
        # Fallback
        cv2.imwrite(backswing_path, midimages[0] if midimages else frame)
        result["backswing"] = backswing_path
    
    if os.path.exists(midswing_path):
        result["midswing"] = midswing_path
    else:
        # Fallback
        mid_idx = len(midimages) // 2
        cv2.imwrite(midswing_path, midimages[mid_idx] if midimages else frame)
        result["midswing"] = midswing_path
    
    if os.path.exists(followthrough_path):
        result["followthrough"] = followthrough_path
    else:
        # Fallback
        cv2.imwrite(followthrough_path, midimages[-1] if midimages else frame)
        result["followthrough"] = followthrough_path
    
    return result 
