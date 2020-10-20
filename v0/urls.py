from django.urls import path, include, re_path
from django.views.static import serve
from django.conf.urls.static import static
from django.conf.urls import url
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.index, name='index'), # Display all the uploaded images
    path('get_table/', views.index_table, name='index_table'),
    path('index_delete', views.index_delete, name='index_delete'),
    path('index_delete_with_local', views.index_delete_with_local, name='index_delete_with_local'),
    path('upload', views.upload, name='upload'), # Upload an image
    path('upload_handle/', views.upload_handle, name='upload_handle'), # process the uploaded image
    path('batch_upload/', views.batch_upload, name='batch_upload'), # Upload an image
    path('batch_upload_handle/', views.batch_upload_handle, name='batch_upload_handle'), # process the uploaded image

    path('process', views.process, name='process'), # Generate masks, keypoints and diameter.
    path('process_handle', views.process_handle, name='process_handle'), # Generate masks, keypoints and diameter.
    path('process_progress', views.process_progress, name='process_progress'),

    path('detail/<slug:name>/', views.detail, name='detail'), # Show the detail of an image
    path('manual_keypoint/<slug:name>/', views.manual_keypoint, name='manual_keypoint'),  # Manual marking of key points
    path('manual_keypoint_handle', views.manual_keypoint_handle, name='manual_keypoint_handle'),
    re_path(r'^IMAGES/(?P<path>.*)$', serve, {'document_root': 'IMAGES'}),


    # re_path(r'^IMAGES/OriginalImages/(?P<path>.*)$', serve, {'document_root': 'IMAGES/OriginalImages'}),
    # re_path(r'^IMAGES/SegmentedImages/(?P<path>.*)$', serve, {'document_root': 'IMAGES/SegmentedImages'}),
    # re_path(r'^IMAGES/KeypointImages/(?P<path>.*)$', serve, {'document_root': 'IMAGES/KeypointImages'}),
    #path('images/<slug:name>/', views.detail, name='detail'), # show the dedail information of a image

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)