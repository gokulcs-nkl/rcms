from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),  # Landing page
    path('login/', views.user_login, name='login'),  # Login page
    path('capture_student/', views.capture_student, name='capture_student'),
    path('home/', views.home, name='home'),
    path('selfie-success/', views.selfie_success, name='selfie_success'),
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('process-face-recognition/', views.process_face_recognition, name='process_face_recognition'),
    path('manual-check-in-out/', views.manual_check_in_out, name='manual_check_in_out'),
    path('students/attendance/', views.student_attendance_list, name='student_attendance_list'),
    path('students/attendance/export/', views.export_attendance_csv, name='export_attendance_csv'),
    path('object-detection-report/', views.object_detection_report, name='object_detection_report'),    path('students/', views.student_list, name='student-list'),
    path('students/<int:pk>/', views.student_detail, name='student-detail'),
    path('students/<int:pk>/authorize/', views.student_authorize, name='student-authorize'),
    path('students/<int:pk>/delete/', views.student_delete, name='student-delete'),
    path('logout/', views.user_logout, name='logout'),
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
]