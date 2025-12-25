from django.db import models
from django.utils import timezone

class Student(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    phone_number = models.CharField(max_length=15)
    student_class = models.CharField(max_length=100)
    image = models.ImageField(upload_to='students/')
    authorized = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-date', '-check_in_time']

    def __str__(self):
        status = "Checked In" if self.check_in_time and not self.check_out_time else "Checked Out" if self.check_out_time else "Pending"
        return f"{self.student.name} - {self.date} - {status}"

    def mark_checked_in(self):
        self.check_in_time = timezone.now()
        self.save()

    def mark_checked_out(self):
        if self.check_in_time:
            self.check_out_time = timezone.now()
            self.save()
        else:
            raise ValueError("Cannot mark check-out without check-in.")

    def calculate_duration(self):
        if self.check_in_time and self.check_out_time:
            duration = self.check_out_time - self.check_in_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        return None

    def save(self, *args, **kwargs):
        if not self.pk:  # Only on creation
            self.date = timezone.now().date()
        super().save(*args, **kwargs)




class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(max_length=255, help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)")
    threshold = models.FloatField(default=0.6, help_text="Face recognition confidence threshold")

    def __str__(self):
        return self.name

class ClassroomObjectDetection(models.Model):
    """Model to track detected prohibited objects in the classroom"""
    OBJECT_TYPES = [
        ('mobile_phone', 'Mobile Phone'),
        ('laptop', 'Laptop'),
        ('tablet', 'Tablet'),
        ('smartwatch', 'Smartwatch'),
        ('headphones', 'Headphones'),
        ('camera', 'Camera'),
        ('other_gadget', 'Other Electronic Gadget'),
    ]
    
    date_detected = models.DateTimeField(auto_now_add=True)
    object_type = models.CharField(max_length=50, choices=OBJECT_TYPES)
    confidence_score = models.FloatField(default=0.0, help_text="Detection confidence (0-1)")
    camera_configuration = models.ForeignKey(CameraConfiguration, on_delete=models.SET_NULL, null=True, blank=True)
    location_x = models.IntegerField(null=True, blank=True, help_text="X coordinate of detected object")
    location_y = models.IntegerField(null=True, blank=True, help_text="Y coordinate of detected object")
    width = models.IntegerField(null=True, blank=True, help_text="Object bounding box width")
    height = models.IntegerField(null=True, blank=True, help_text="Object bounding box height")
    image_snapshot = models.ImageField(upload_to='detections/', null=True, blank=True, help_text="Snapshot of detected object")
    notes = models.TextField(blank=True, help_text="Additional notes about the detection")

    class Meta:
        ordering = ['-date_detected']

    def __str__(self):
        return f"{self.get_object_type_display()} detected at {self.date_detected.strftime('%Y-%m-%d %H:%M:%S')}"