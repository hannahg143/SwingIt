import os
import sys
import json
import tempfile
import asyncio

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['HOME'] = '/tmp'  

if 'LAMBDA_TASK_ROOT' in os.environ:
    lambda_task_root = os.environ['LAMBDA_TASK_ROOT']
    
    import glob
    import ctypes
    import ctypes.util
    
    libs_dirs = []
    
    for libs_dir in glob.glob(os.path.join(lambda_task_root, '*.libs')):
        if os.path.isdir(libs_dir):
            libs_dirs.append(libs_dir)
    
    for root, dirs, files in os.walk(lambda_task_root):
        if '.libs' in dirs:
            libs_path = os.path.join(root, '.libs')
            if libs_path not in libs_dirs:
                libs_dirs.append(libs_path)
    
    if libs_dirs:
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(libs_dirs)
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
        
        # Load dependencies first before OpenBLAS
        numpy_libs_dir = os.path.join(lambda_task_root, 'numpy.libs')
        if os.path.exists(numpy_libs_dir):
            try:
                all_libs = os.listdir(numpy_libs_dir)
                print(f"Files in numpy.libs: {all_libs}")
            except Exception as e:
                print(f"Could not list numpy.libs directory: {e}")
            
            
            libquadmath_loaded = False
            libgfortran_loaded = False
            
            libquadmath_files = glob.glob(os.path.join(numpy_libs_dir, 'libquadmath*.so*'))
            for libquadmath_path in libquadmath_files:
                if os.path.isfile(libquadmath_path):
                    try:
                        abs_path = os.path.abspath(libquadmath_path)
                        ctypes.CDLL(abs_path, mode=ctypes.RTLD_GLOBAL)
                        print(f"Successfully preloaded libquadmath from {abs_path}")
                        libquadmath_loaded = True
                        break  # Only need to load one
                    except Exception as e:
                        print(f"Failed to preload libquadmath {libquadmath_path}: {type(e).__name__}: {e}")
            
            if libquadmath_loaded:
                libgfortran_path = None
                exact_path = os.path.join(numpy_libs_dir, 'libgfortran-040039e1.so.5.0.0')
                if os.path.exists(exact_path):
                    libgfortran_path = exact_path
                else:
                    libgfortran_files = glob.glob(os.path.join(numpy_libs_dir, 'libgfortran*.so*'))
                    if libgfortran_files:
                        libgfortran_path = libgfortran_files[0]
                        print(f"Found libgfortran via glob: {libgfortran_path}")
                
                if libgfortran_path and os.path.exists(libgfortran_path):
                    try:
                        abs_path = os.path.abspath(libgfortran_path)
                        ctypes.CDLL(abs_path, mode=ctypes.RTLD_GLOBAL)
                        print(f"Successfully preloaded libgfortran from {abs_path}")
                        libgfortran_loaded = True
                    except Exception as e:
                        print(f"Failed to preload libgfortran {libgfortran_path}: {type(e).__name__}: {e}")
                else:
                    print(f"libgfortran not found in {numpy_libs_dir}")
            else:
                print(f"Skipping libgfortran, libquadmath not loaded")
            
            if libgfortran_loaded:
                openblas_files = glob.glob(os.path.join(numpy_libs_dir, 'libopenblas*.so*'))
                for openblas_path in openblas_files:
                    if os.path.isfile(openblas_path):
                        try:
                            abs_path = os.path.abspath(openblas_path)
                            ctypes.CDLL(abs_path, mode=ctypes.RTLD_GLOBAL)
                            print(f"Successfully preloaded OpenBLAS from {abs_path}")
                            break  # Only need to load one
                        except Exception as e:
                            print(f"Failed to preload OpenBLAS {openblas_path}: {type(e).__name__}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                print(f"Skipping OpenBLAS, libgfortran not loaded")
            
            try:
                import shutil
                tmp_lib_dir = '/tmp/lib'
                os.makedirs(tmp_lib_dir, exist_ok=True)
                
                lib_mappings = {
                    'libgfortran.so.5': 'libgfortran-040039e1.so.5.0.0',
                    'libgfortran.so': 'libgfortran-040039e1.so.5.0.0',
                    'libquadmath.so.0': 'libquadmath-96973f99.so.0.0.0',
                    'libquadmath.so': 'libquadmath-96973f99.so.0.0.0'
                }
                for std_name, hash_name in lib_mappings.items():
                    source_path = os.path.join(numpy_libs_dir, hash_name)
                    dest_path = os.path.join(tmp_lib_dir, std_name)
                    if os.path.exists(source_path) and not os.path.exists(dest_path):
                        try:
                            shutil.copy2(source_path, dest_path)
                            print(f"Copied {hash_name} to {dest_path}")
                        except Exception as e:
                            print(f"Failed to copy {hash_name}: {e}")
                
                for lib_file in all_libs:
                    if lib_file.endswith('.so') or '.so.' in lib_file:
                        source_path = os.path.join(numpy_libs_dir, lib_file)
                        dest_path = os.path.join(tmp_lib_dir, lib_file)
                        if os.path.exists(source_path) and not os.path.exists(dest_path):
                            try:
                                shutil.copy2(source_path, dest_path)
                            except Exception as e:
                                pass 
                
                current_ld = os.environ.get('LD_LIBRARY_PATH', '')
                if tmp_lib_dir not in current_ld:
                    os.environ['LD_LIBRARY_PATH'] = f"{tmp_lib_dir}:{current_ld}" if current_ld else tmp_lib_dir
            except Exception as e:
                print(f"Could not copy libraries to /tmp/lib: {e}")
                import traceback
                traceback.print_exc()
    
    numpy_dir = os.path.join(lambda_task_root, 'numpy')
    if os.path.exists(numpy_dir):
        print(f"numpy directory found at {numpy_dir}")
        print(f"Has __init__.py: {os.path.exists(os.path.join(numpy_dir, '__init__.py'))}")
        print(f"Has __config__.py: {os.path.exists(os.path.join(numpy_dir, '__config__.py'))}")
        print(f"Has setup.py: {os.path.exists(os.path.join(numpy_dir, 'setup.py'))}")
    else:
        print(f"numpy directory NOT found at {numpy_dir}")

import boto3
from botocore.exceptions import ClientError

try:
    import numpy as np
    print("numpy imported successfully")
except Exception as e:
    print(f"numpy import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise

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

def lambda_handler(event, context):
    try:
        # Get AWS configuration from environment variables
        s3_bucket = os.environ.get("S3_BUCKET_NAME")

        aws_region = os.environ.get("AWS_REGION") or context.invoked_function_arn.split(":")[3] if context else "us-east-1"
        
        if not s3_bucket:
            raise ValueError("S3_BUCKET_NAME environment variable required")

        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        
        file_key = event["fileId"]  # S3 key of the video file
        user_id = event.get("userId", "user")
        backswing_id = event.get("backswingId", f"{user_id}_backswing")
        midswing_id = event.get("midswingId", f"{user_id}_midswing")
        followthrough_id = event.get("followthroughId", f"{user_id}_followthrough")

        s3_client = boto3.client('s3', region_name=aws_region)

        # Download video from S3
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input.mp4")
        
        s3_client.download_file(s3_bucket, file_key, video_path)

        # Run analysis
        key_frames = asyncio.run(run_forehand_analysis(video_path, temp_dir, user_id))

        # Upload results to S3
        result_keys = {}
        id_map = {"backswing": backswing_id, "midswing": midswing_id, "followthrough": followthrough_id}
        
        for k, path in key_frames.items():
            if path and os.path.exists(path):
                custom_file_key = id_map.get(k, f"{user_id}_{k}")
                print(f"Uploading {k} with key: {custom_file_key}")

                # Upload to S3
                s3_client.upload_file(
                    path,
                    s3_bucket,
                    custom_file_key,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
                
                # Generate presigned URL
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': s3_bucket, 'Key': custom_file_key},
                    ExpiresIn=3600
                )
                result_keys[k] = {
                    "key": custom_file_key,
                    "url": presigned_url
                }

        response_data = {
            "success": True,
            "frames": result_keys
        }
        
        print(f"Function completed successfully")
        print(f"Result keys: {result_keys}")
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(response_data)
        }
    
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e)
        }
        
        print(f"Function failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(error_response)
        }

async def run_forehand_analysis(video_path, temp_dir, name):
    if not cv2_available:
        print("OpenCV not available, cannot process video")
        return {}
    
    await asyncio.sleep(0.01)
    
    if mp_available:
        
        try:
            mp_pose = mp.solutions.pose
            
            # Model should be pre-downloaded during Docker build
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
            import traceback
            traceback.print_exc()
            use_mediapipe = False
            pose = None
            mp_drawing = None
            mp_pose = None
    else:
        use_mediapipe = False
        pose = None
        mp_drawing = None
        mp_pose = None
        print("MediaPipe not available, using fallback detection")
    
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
            right_wrist_x = landmarks[16].x  # Right wrist
            
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
            right_wrist_x = landmarks[16].x  # Right wrist 
            
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
            right_wrist_x = landmarks[16].x     # Right wrist

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
        if use_mediapipe and pose:
            # MediaPipe version
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                return (results.pose_landmarks.landmark, results.pose_landmarks)
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
    all_frames = []  # Store all processed frames
    all_wrist_x = []  # Store wrist x positions for each frame
    all_middist = []  # Store mid distances for each frame
    
    MIN_START_FRAME = 10  # Skip first 10 frames

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
            wrist_x = None
            middist = None
            
            if detection_data is not None:
                if use_mediapipe:
                    landmarks, pose_landmarks_obj = detection_data

                    mp_drawing.draw_landmarks(
                        image, 
                        pose_landmarks_obj,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Get wrist x position
                    wrist_x = landmarks[16].x  # Right wrist
                    
                    # Get mid distance (wrist to hip)
                    middist = findMidF(landmarks)
                    
                    # Add frame info overlay
                    cv2.putText(image, f"Frame {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(image, f"Wrist X: {wrist_x:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Fallback, no wrist tracking
                    motion_score = detection_data
                    cv2.putText(image, f"Frame {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(image, f"Motion: {motion_score:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # No detection
                cv2.putText(image, f"Frame {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, "No Pose Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
        except Exception as e:
            print(f"Error in pose detection at frame {count}: {e}")
            image = frame
            wrist_x = None
            middist = None
        
        # Store frame data
        all_frames.append(image)
        all_wrist_x.append(wrist_x)
        all_middist.append(middist)
        
        if count % 50 == 0:
            print(f"Processing frame {count}, wrist_x={wrist_x}")
        
        count += 1
        await asyncio.sleep(0.001)
            
    cap.release()
    
    if use_mediapipe and pose:
        pose.close()
    
    valid_wrist_count = sum(1 for x in all_wrist_x if x is not None)
    
    if valid_wrist_count == 0:
        print("No valid wrist positions detected")
        # Return fallback
        result = {}
        backswing_path = os.path.join(temp_dir, f"backswing_{name}.jpg")
        midswing_path = os.path.join(temp_dir, f"midswing_{name}.jpg")
        followthrough_path = os.path.join(temp_dir, f"followthrough_{name}.jpg")
        cv2.imwrite(backswing_path, all_frames[0] if all_frames else frame)
        cv2.imwrite(midswing_path, all_frames[len(all_frames)//2] if all_frames else frame)
        cv2.imwrite(followthrough_path, all_frames[-1] if all_frames else frame)
        result["backswing"] = backswing_path
        result["midswing"] = midswing_path
        result["followthrough"] = followthrough_path
        return result
    
    # Print sample wrist_x values for debugging
    valid_positions = [(i, x) for i, x in enumerate(all_wrist_x) if x is not None]
    print(f"Sample wrist_x positions:")
    for i in range(0, len(valid_positions), max(1, len(valid_positions)//10)):
        frame_num, wrist_val = valid_positions[i]
        print(f"  Frame {frame_num}: wrist_x = {wrist_val:.4f}")
    
    # Find backswing: frame with minimum wrist_x (furthest left on screen)
    # Find follow-through: frame with maximum wrist_x (furthest right on screen)
    # Only consider frames after MIN_START_FRAME
    
    backframe = -1
    followframe = -1
    min_wrist_x = float('inf')
    max_wrist_x = float('-inf')
    
    for i in range(MIN_START_FRAME, len(all_wrist_x)):
        if all_wrist_x[i] is not None:
            if all_wrist_x[i] < min_wrist_x:
                min_wrist_x = all_wrist_x[i]
                backframe = i
            if all_wrist_x[i] > max_wrist_x:
                max_wrist_x = all_wrist_x[i]
                followframe = i
    
    
    if backframe > followframe:
        print(f"Backswing frame ({backframe}) comes AFTER follow-through frame ({followframe})")
    
    # Find midswing: frame between backswing and followthrough with minimum distance to hip (contact point)
    midframe = -1
    min_middist = float('inf')
    
    # Search between the two key frames (chronologically)
    start_search = min(backframe, followframe)
    end_search = max(backframe, followframe)
    
    for i in range(start_search, end_search + 1):
        if all_middist[i] is not None and all_middist[i] < min_middist:
            min_middist = all_middist[i]
            midframe = i
    
    result = {}
    backswing_path = os.path.join(temp_dir, f"backswing_{name}.jpg")
    midswing_path = os.path.join(temp_dir, f"midswing_{name}.jpg")
    followthrough_path = os.path.join(temp_dir, f"followthrough_{name}.jpg")
    
    # Save backswing frame
    if backframe != -1 and backframe < len(all_frames):
        back_img = all_frames[backframe].copy()
        cv2.putText(back_img, "BACKSWING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(backswing_path, back_img)
        result["backswing"] = backswing_path
    else:
        # Fallback to first frame
        cv2.imwrite(backswing_path, all_frames[0] if all_frames else frame)
        result["backswing"] = backswing_path
    
    # Save midswing frame
    if midframe != -1 and midframe < len(all_frames):
        mid_img = all_frames[midframe].copy()
        cv2.putText(mid_img, "MIDSWING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(midswing_path, mid_img)
        result["midswing"] = midswing_path
    else:
        # Fallback to middle frame
        mid_idx = len(all_frames) // 2
        cv2.imwrite(midswing_path, all_frames[mid_idx] if all_frames else frame)
        result["midswing"] = midswing_path
    
    # Save follow-through frame
    if followframe != -1 and followframe < len(all_frames):
        follow_img = all_frames[followframe].copy()
        cv2.putText(follow_img, "FOLLOW-THROUGH", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(followthrough_path, follow_img)
        result["followthrough"] = followthrough_path
    else:
        # Fallback to last frame
        cv2.imwrite(followthrough_path, all_frames[-1] if all_frames else frame)
        result["followthrough"] = followthrough_path
    
    return result 
