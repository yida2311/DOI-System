from django.contrib import admin
from .models import Images, Masks
# Register your models here.
admin.site.register(Images)
admin.site.register(Masks)
#admin.site.register(Keypoint_masks)

