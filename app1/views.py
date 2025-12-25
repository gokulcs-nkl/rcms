import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration, ClassroomObjectDetection
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
import csv
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Student
from django.views.decorators.csrf import csrf_exempt
import json
from .object_detection import get_object_detector, detect_prohibited_items
from django.db import models

# Cache for known face encodings to speed up recognition
KNOWN_FACE_ENCODINGS = None
KNOWN_FACE_NAMES = None
LAST_KNOWN_LOAD = 0
KNOWN_CACHE_TTL = 60  # seconds


# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces (can return multiple)
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

def get_known_faces(force_reload=False):
    """Load and cache known faces to speed up recognition."""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES, LAST_KNOWN_LOAD

    now = time.time()
    if force_reload or KNOWN_FACE_ENCODINGS is None or (now - LAST_KNOWN_LOAD) > KNOWN_CACHE_TTL:
        known_face_encodings = []
        known_face_names = []

        uploaded_images = Student.objects.filter(authorized=True)
        for student in uploaded_images:
            image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
            if not os.path.exists(image_path):
                continue
            known_image = cv2.imread(image_path)
            if known_image is None:
                continue
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.extend(encodings)
                known_face_names.extend([student.name] * len(encodings))

        KNOWN_FACE_ENCODINGS = np.array(known_face_encodings) if known_face_encodings else None
        KNOWN_FACE_NAMES = known_face_names if known_face_names else None
        LAST_KNOWN_LOAD = now

    return KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES

# Backward compatible wrapper
def encode_uploaded_images():
    return get_known_faces(force_reload=False)

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# Landing page view
def landing(request):
    return render(request, 'landing.html')

# View for capturing student information and image
def capture_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        student_class = request.POST.get('student_class')
        image_data = request.POST.get('image_data')

        # Decode the base64 image data
        if image_data:
            header, encoded = image_data.split(',', 1)
            image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

            student = Student(
                name=name,
                email=email,
                phone_number=phone_number,
                student_class=student_class,
                image=image_file,
                authorized=False  # Default to False during registration
            )
            student.save()

            return redirect('selfie_success')  # Redirect to a success page

    return render(request, 'capture_student.html')


# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')


# This views for capturing studen faces and recognize
def capture_and_recognize(request):
    # If GET request, just show the page
    if request.method == 'GET':
        return render(request, 'capture_and_recognize.html')
    
    # If POST request, start the face recognition process
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        object_detector = get_object_detector()  # Get object detector instance
        
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  # Use integer index for webcam
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  # Use string for IP camera URL

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  # load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)  # Function to detect and encode face in frame

                # ===== OBJECT DETECTION =====
                # Detect prohibited items/gadgets in the classroom
                try:
                    detected_objects, annotated_frame = detect_prohibited_items(frame)
                    
                    if detected_objects:
                        # Log detected objects to database
                        for obj in detected_objects:
                            try:
                                object_type_key, _ = object_detector.get_object_type_key(obj['class'])
                                ClassroomObjectDetection.objects.create(
                                    object_type=object_type_key,
                                    confidence_score=obj['confidence'],
                                    camera_configuration=cam_config,
                                    location_x=obj['bbox'][0],
                                    location_y=obj['bbox'][1],
                                    width=obj['width'],
                                    height=obj['height'],
                                    notes=f"Detected: {obj['class']} in classroom"
                                )
                                print(f"✓ Logged: {obj['class']} with confidence {obj['confidence']:.2f}")
                            except Exception as db_error:
                                print(f"✗ Error saving detection to database: {db_error}")
                        
                        # Use annotated frame with object detections
                        frame = annotated_frame
                    else:
                        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                except Exception as obj_detect_error:
                    print(f"✗ Object detection error: {obj_detect_error}")
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # ===== FACE RECOGNITION =====
                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings is not None:
                        boxes, _ = mtcnn.detect(frame_rgb)
                        if boxes is None:
                            boxes = []

                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        for name, box in zip(names, boxes):
                            if box is None:
                                continue
                            (x1, y1, x2, y2) = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                            if name != 'Not Recognized':
                                students = Student.objects.filter(name=name)
                                if students.exists():
                                    student = students.first()

                                    # Check proximity - face should be reasonably close
                                    # Calculate face size to estimate distance
                                    face_width = x2 - x1
                                    face_height = y2 - y1
                                    face_area = face_width * face_height
                                    frame_area = frame.shape[0] * frame.shape[1]
                                    face_ratio = face_area / frame_area
                                    
                                    # Face should occupy at least 2% of frame for valid attendance
                                    is_close_enough = face_ratio > 0.02
                                    
                                    if not is_close_enough:
                                        cv2.putText(frame, f"{name}, come closer!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                                    else:
                                        # Manage attendance based on check-in and check-out logic with re-entry support
                                        attendance, created = Attendance.objects.get_or_create(student=student, date=datetime.now().date())
                                        
                                        if created:
                                            # First time today - check in
                                            attendance.mark_checked_in()
                                            success_sound.play()
                                            cv2.putText(frame, f"{name}, checked in.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        else:
                                            # Already has record today
                                            if attendance.check_in_time and not attendance.check_out_time:
                                                # Currently checked in - allow checkout after 60 seconds
                                                time_since_checkin = timezone.now() - attendance.check_in_time
                                                if time_since_checkin >= timedelta(seconds=60):
                                                    attendance.mark_checked_out()
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    remaining = 60 - int(time_since_checkin.total_seconds())
                                                    cv2.putText(frame, f"{name}, already in. ({remaining}s)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                            elif attendance.check_in_time and attendance.check_out_time:
                                                # Already checked out today - allow re-entry
                                                time_since_checkout = timezone.now() - attendance.check_out_time
                                                if time_since_checkout >= timedelta(seconds=300):  # 5 minutes before re-entry
                                                    # Create new attendance record for re-entry
                                                    new_attendance = Attendance.objects.create(
                                                        student=student,
                                                        date=datetime.now().date()
                                                    )
                                                    new_attendance.mark_checked_in()
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, re-entered!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    remaining = 300 - int(time_since_checkout.total_seconds())
                                                    cv2.putText(frame, f"{name}, checked out. Wait {remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  # Check if window exists
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    # After processing, send user back to home instead of attendance list
    return redirect('home')


@csrf_exempt
def process_face_recognition(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

    try:
        data = json.loads(request.body)
        image_data = data.get('image', '')

        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image data'})

        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({'status': 'error', 'message': 'Invalid image'})

        # Convert BGR to RGB and optionally downscale large frames for speed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if max(h, w) > 720:
            scale = 720.0 / max(h, w)
            frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

        # Detect and encode faces in the frame
        test_face_encodings = detect_and_encode(frame_rgb)
        if not test_face_encodings:
            return JsonResponse({'status': 'no_face', 'message': 'No face detected'})

        # Get cached known faces
        known_face_encodings, known_face_names = get_known_faces(force_reload=False)
        if known_face_encodings is None or known_face_names is None:
            return JsonResponse({'status': 'error', 'message': 'No authorized students found'})

        names = recognize_faces(known_face_encodings, known_face_names, test_face_encodings, threshold=0.6)

        results = []
        for recognized_name in names:
            if recognized_name == 'Not Recognized':
                results.append({'name': recognized_name, 'action': 'Not Recognized'})
                continue

            student = Student.objects.filter(name=recognized_name).first()
            if not student:
                results.append({'name': recognized_name, 'action': 'Not Recognized'})
                continue

            attendance, created = Attendance.objects.get_or_create(
                student=student,
                date=datetime.now().date()
            )

            if created:
                attendance.mark_checked_in()
                results.append({'name': recognized_name, 'action': 'Checked In', 'time': str(attendance.check_in_time)})
            else:
                if attendance.check_in_time and not attendance.check_out_time:
                    if timezone.now() >= attendance.check_in_time + timedelta(seconds=60):
                        attendance.mark_checked_out()
                        results.append({'name': recognized_name, 'action': 'Checked Out', 'time': str(attendance.check_out_time)})
                    else:
                        results.append({'name': recognized_name, 'action': 'Already Checked In', 'time': str(attendance.check_in_time)})
                elif attendance.check_in_time and attendance.check_out_time:
                    results.append({'name': recognized_name, 'action': 'Already Checked Out', 'time': str(attendance.check_out_time)})

        any_success = any(r.get('action') not in ['Not Recognized'] for r in results)
        if any_success:
            return JsonResponse({'status': 'success', 'results': results})
        else:
            return JsonResponse({'status': 'not_recognized', 'results': results, 'message': 'Faces not recognized'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


#this is for showing Attendance list
def student_attendance_list(request):
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all students
    students = Student.objects.all()

    # Filter students based on the search query
    if search_query:
        students = students.filter(name__icontains=search_query)

    # Prepare the attendance data
    student_attendance_data = []

    for student in students:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student': student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,  # Pass the search query to the template
        'date_filter': date_filter       # Pass the date filter to the template
    }
    return render(request, 'student_attendance_list.html', context)


def home(request):
    # If not logged in, redirect to login page
    if not request.user.is_authenticated:
        return redirect('login')
    
    import json
    from django.core.serializers.json import DjangoJSONEncoder
    
    total_students = Student.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_time__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_time__isnull=False).count()
    total_cameras = CameraConfiguration.objects.count()

    # Prepare attendance data grouped by class
    students = Student.objects.all()
    student_data = []
    
    for student in students:
        attendance_records = Attendance.objects.filter(student=student)
        records = []
        for record in attendance_records:
            records.append({
                'id': record.id,
                'date': str(record.date) if record.date else None,
                'check_in_time': str(record.check_in_time) if record.check_in_time else None,
                'check_out_time': str(record.check_out_time) if record.check_out_time else None
            })
        
        student_data.append({
            'student': {
                'id': student.id,
                'name': student.name,
                'student_class': student.student_class if student.student_class else 'Unknown'
            },
            'attendance_records': records
        })
    
    context = {
        'total_students': total_students,
        'total_attendance': total_attendance,
        'total_check_ins': total_check_ins,
        'total_check_outs': total_check_outs,
        'total_cameras': total_cameras,
        'student_attendance_data': json.dumps(student_data),
    }
    return render(request, 'home.html', context)


# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students})

@login_required
@user_passes_test(is_admin)
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})

@login_required
@user_passes_test(is_admin)
def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('student-detail', pk=pk)
    
    return render(request, 'student_authorize.html', {'student': student})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list')  # Redirect to the student list after deletion
    
    return render(request, 'student_delete_confirm.html', {'student': student})


# View function for user login
def user_login(request):
    # If already logged in, go straight to home
    if request.user.is_authenticated:
        # Check if there's a next parameter
        next_url = request.GET.get('next', 'home')
        return redirect(next_url)

    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect to next URL if provided, otherwise go to home
            next_url = request.POST.get('next') or request.GET.get('next', 'home')
            return redirect(next_url)
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Pass the next parameter to the template for GET requests
    context = {
        'next': request.GET.get('next', '')
    }
    return render(request, 'login.html', context)


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('landing')  # Redirect to landing page after logout

# View to display detected objects
@login_required
def object_detection_report(request):
    """Display all detected prohibited objects in the classroom"""
    detections = ClassroomObjectDetection.objects.all().order_by('-date_detected')
    
    # Filter by object type if provided
    object_type = request.GET.get('object_type', None)
    if object_type:
        detections = detections.filter(object_type=object_type)
    
    # Filter by date range if provided
    from_date = request.GET.get('from_date', None)
    to_date = request.GET.get('to_date', None)
    if from_date:
        detections = detections.filter(date_detected__gte=from_date)
    if to_date:
        detections = detections.filter(date_detected__lte=to_date)
    
    # Summary statistics
    total_detections = ClassroomObjectDetection.objects.count()
    object_type_summary = ClassroomObjectDetection.objects.values('object_type').annotate(count=models.Count('id'))
    
    context = {
        'detections': detections,
        'total_detections': total_detections,
        'object_type_summary': object_type_summary,
        'object_types': ClassroomObjectDetection.OBJECT_TYPES,
    }
    
    return render(request, 'object_detection_report.html', context)

# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()  

        # Redirect to the list page after successful update
        return redirect('camera_config_list')  
    
    # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()  
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})


@login_required
def manual_check_in_out(request):
    """Manual check-in/check-out interface for students"""
    try:
        students = Student.objects.filter(authorized=True).order_by('name')
        message = None
        message_type = 'info'
        
        if request.method == 'POST':
            student_id = request.POST.get('student_id')
            action = request.POST.get('action')
            
            try:
                student = Student.objects.get(id=student_id, authorized=True)
                today = datetime.now().date()
                
                if action == 'check_in':
                    # Check if already checked in today
                    existing_attendance = Attendance.objects.filter(
                        student=student, 
                        date=today,
                        check_in_time__isnull=False,
                        check_out_time__isnull=True
                    ).first()
                    
                    if existing_attendance:
                        message = f"{student.name} is already checked in."
                        message_type = 'warning'
                    else:
                        # Create new attendance record
                        attendance = Attendance.objects.create(student=student, date=today)
                        attendance.mark_checked_in()
                        message = f"{student.name} successfully checked in at {attendance.check_in_time.strftime('%I:%M %p')}."
                        message_type = 'success'
                
                elif action == 'check_out':
                    # Find the most recent check-in without check-out
                    attendance = Attendance.objects.filter(
                        student=student,
                        date=today,
                        check_in_time__isnull=False,
                        check_out_time__isnull=True
                    ).order_by('-check_in_time').first()
                    
                    if attendance:
                        attendance.mark_checked_out()
                        duration = attendance.calculate_duration()
                        message = f"{student.name} successfully checked out at {attendance.check_out_time.strftime('%I:%M %p')}. Duration: {duration}"
                        message_type = 'success'
                    else:
                        message = f"{student.name} is not checked in yet today."
                        message_type = 'warning'
                        
            except Student.DoesNotExist:
                message = "Student not found or not authorized."
                message_type = 'error'
            except Exception as e:
                message = f"Error: {str(e)}"
                message_type = 'error'
        
        # Get today's attendance status for all students
        today = datetime.now().date()
        student_status = []
        for student in students:
            latest_attendance = Attendance.objects.filter(
                student=student,
                date=today
            ).order_by('-check_in_time').first()
            
            status = 'not_checked_in'
            status_text = 'Not Checked In'
            last_action_time = None
            
            if latest_attendance:
                if latest_attendance.check_in_time and not latest_attendance.check_out_time:
                    status = 'checked_in'
                    status_text = 'Checked In'
                    last_action_time = latest_attendance.check_in_time
                elif latest_attendance.check_in_time and latest_attendance.check_out_time:
                    status = 'checked_out'
                    status_text = 'Checked Out'
                    last_action_time = latest_attendance.check_out_time
            
            student_status.append({
                'student': student,
                'status': status,
                'status_text': status_text,
                'last_action_time': last_action_time
            })
        
        context = {
            'student_status': student_status,
            'message': message,
            'message_type': message_type
        }
        
        return render(request, 'manual_check_in_out.html', context)
    
    except Exception as e:
        # Log the error and show a user-friendly error page
        print(f"Error in manual_check_in_out: {str(e)}")
        return render(request, 'error.html', {'error_message': f'An error occurred: {str(e)}'})


def export_attendance_csv(request):
    """Export attendance records to CSV based on filters."""
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all students
    students = Student.objects.all()

    # Filter students based on the search query
    if search_query:
        students = students.filter(name__icontains=search_query)

    # Create the HttpResponse object with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.csv"'

    # Create the CSV writer object
    writer = csv.writer(response)
    
    # Write the header row
    writer.writerow(['Student Name', 'Student Email', 'Student Class', 'Attendance Date', 'Check-In Time', 'Check-Out Time', 'Duration'])

    # Prepare and write the attendance data
    for student in students:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')

        for attendance in attendance_records:
            # Calculate duration if both check-in and check-out times exist
            duration = ''
            if attendance.check_in_time and attendance.check_out_time:
                duration = str(attendance.calculate_duration())
            else:
                duration = 'Not Completed'

            # Write the attendance record
            writer.writerow([
                student.name,
                student.email,
                student.student_class,
                attendance.date,
                attendance.check_in_time.strftime('%H:%M:%S') if attendance.check_in_time else 'N/A',
                attendance.check_out_time.strftime('%H:%M:%S') if attendance.check_out_time else 'N/A',
                duration
            ])

    return response
    return render(request, 'camera_config_delete.html', {'config': config})
