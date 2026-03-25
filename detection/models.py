from django.db import models

class Ad(models.Model):
    media = models.FileField(upload_to='ads/', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

# models.py
from django.db import models

class ClickEvent(models.Model):
    # ================= USER / DEVICE =================
    device_type = models.CharField(max_length=50)   # mobile / desktop
    browser = models.CharField(max_length=200)
    operating_system = models.CharField(max_length=100)
    referrer_url = models.TextField()
    page_url = models.TextField()

    # ================= BEHAVIOR FEATURES =================
    click_duration = models.FloatField()             # seconds
    mouse_movement = models.FloatField()             # count
    scroll_depth = models.FloatField()               # 0–1
    keystrokes_detected = models.IntegerField()
    ad_position = models.IntegerField()
    time_since_last_click = models.FloatField()      # seconds
    click_frequency = models.IntegerField()           # clicks in window
    VPN_usage = models.BooleanField(default=False)

    # ================= MODEL OUTPUT =================
    fraud_probability = models.FloatField(null=True, blank=True)
    fraud_label = models.CharField(
        max_length=10,
        choices=[("fraud", "Fraud"), ("legit", "Legit")],
        null=True,
        blank=True
    )

    # ================= METADATA =================
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.device_type} | {self.fraud_label} | {self.timestamp}"

class TrainingMetrics(models.Model):
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    trained_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Training at {self.trained_at} - Acc: {self.accuracy}"
