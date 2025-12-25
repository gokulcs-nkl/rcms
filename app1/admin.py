from django.contrib import admin
from .models import Student, Attendance, CameraConfiguration, ClassroomObjectDetection

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone_number', 'student_class', 'authorized']
    list_filter = ['student_class', 'authorized']
    search_fields = ['name', 'email']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'check_in_time', 'check_out_time']
    list_filter = ['date']
    search_fields = ['student__name']

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return ['student', 'date', 'check_in_time', 'check_out_time']
        else:  # Adding a new object
            return ['date', 'check_in_time', 'check_out_time']

    def save_model(self, request, obj, form, change):
        if change:  # Editing an existing object
            # Ensure check-in and check-out times cannot be modified via admin
            obj.check_in_time = Attendance.objects.get(id=obj.id).check_in_time
            obj.check_out_time = Attendance.objects.get(id=obj.id).check_out_time
        super().save_model(request, obj, form, change)


@admin.register(CameraConfiguration)
class CameraConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_source', 'threshold']
    search_fields = ['name']


@admin.register(ClassroomObjectDetection)
class ClassroomObjectDetectionAdmin(admin.ModelAdmin):
    list_display = ['object_type', 'date_detected', 'confidence_score', 'camera_configuration']
    list_filter = ['object_type', 'date_detected', 'camera_configuration']
    search_fields = ['notes']
    readonly_fields = ['date_detected', 'confidence_score', 'location_x', 'location_y', 'width', 'height']
    
    fieldsets = (
        ('Detection Information', {
            'fields': ('object_type', 'date_detected', 'confidence_score', 'camera_configuration')
        }),
        ('Location Details', {
            'fields': ('location_x', 'location_y', 'width', 'height')
        }),
        ('Snapshot', {
            'fields': ('image_snapshot',)
        }),
        ('Additional Notes', {
            'fields': ('notes',)
        }),
    )
