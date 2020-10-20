from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import Images, Masks
from PIL import Image
from .toolbox.utils import tracker, get_foot_point, DateEncoder, cut_string_length, judge_tumor_stage
from .toolbox.Get_tumor_diameter import get_tumor_diameter
from .toolbox.filter.filters_apply import apply_color_filter_to_image
from .toolbox.tile.tile import Cutter
from .toolbox.key_map import KeyMap
from .segmentation.run_segmentation import Segmentation
from .segmentation.configs.config import config_django
import os, re, json, time, sys, math
sys.path.append("..")

t = tracker() # Represents the progress of processing.

# Create your views here.
def index(request):
    '''
    The home page. 
    It displays all origin images, including name, tumour stage,
    and history stage.
    '''
    image_list = Images.objects.order_by('-time')
    #template = loader.get_template('index.html')
    template = loader.get_template('table.html')
    context = {
        'image_list': image_list
    }
    return HttpResponse(template.render(context, request))

@csrf_exempt
def index_table(request):
    '''
    Process a request for table data from the main page.
    '''
    image_list = Images.objects.order_by('-time')
    print('----------------')
    data = []
    for image in image_list:
        data.append({
            "name": image.name,
            "stage": image.tumor_stage,
            "depth": str(round(image.depth, 2)) + ' mm',
            "diameter": str(round(image.diameter, 2)) + ' mm',
            "comment": image.comment,
            "time": image.time,
            "description": cut_string_length(image.description, 15)
        })
    return HttpResponse(json.dumps(data, cls = DateEncoder))

@csrf_exempt
def index_delete(request):
    to_delete_name = json.loads(request.body)
    for name in to_delete_name:
        image = Images.objects.get(name=name)
        image.delete()
    print(to_delete_name)
    return JsonResponse('成功删除', safe=False)


@csrf_exempt
def index_delete_with_local(request):
    to_delete_name = json.loads(request.body)
    for name in to_delete_name:
        image = Images.objects.get(name=name)
        image.delete()
    print(to_delete_name)
    return JsonResponse('成功删除', safe=False)


def upload(request):
    '''
    Upload an original image.
    '''
    return render(request, 'fileinput.html')

@csrf_exempt
def upload_handle(request):
    '''
    1. Save the uploaded image.
    2. Apply image segmentation to the uploaded image.
    3. Find out key points.
    '''
    if request.method == 'POST':
        # Save image to the database.
        image = request.FILES.get('image', '')
        print(image)
        description = request.POST.get('description', '')

        image_name = image.name
        existed_image_num = Images.objects.filter(Q(name__startswith=image_name.split('.')[0])).count()

        if existed_image_num: # An image with a duplicate name already exists
            image_name_prefix = image_name.split('.')[0] # xxx
            image_name_postfix = image_name.split('.')[1] # png
            image_name = image_name_prefix + '(' + str(existed_image_num) + ').' + image_name_postfix # xxx(1).png

        # Save image.
        imagefile = '%s/IMAGES/Images/%s'%(settings.BASE_DIR,image_name)
        with open(imagefile, 'wb') as f:
            for c in image.chunks():
                f.write(c)
                
        # Get height and width of image.
        pil_img = Image.open(imagefile)
        width = pil_img.size[0]
        height = pil_img.size[1]

        # Save image into database
        original_image = Images.objects.create(
            image = '/IMAGES/Images/%s' % image_name,
            description = description,
            name = image_name.split('.')[0],
            postfix = image_name.split('.')[1],
            height = height,
            width = width,
            depth = 0,
        )
        original_image.save()


        mask = Masks.objects.create(related_image=original_image, 
                                    name=image_name.split('.')[0],
                                    height = height,
                                    width = width)
        mask.save()
        imagefile = '%s/IMAGES/Masks/%s'%(settings.BASE_DIR,image_name)
        with open(imagefile, 'wb') as f:
            for c in image.chunks():
                f.write(c)



        #return render(request, 'import_success.html')
        return JsonResponse({'success':0})
    return HttpResponse('未上传成功')

def batch_upload(request):
    return render(request, 'batch_fileinput.html')

@csrf_exempt
def batch_upload_handle(request):
    image_names = json.loads(request.body)

    track_total = 3*len(image_names)
    track_count = 0
    t.refresh(track_count) # record the progress of upload
    track_count += 1
    for i,image_name in enumerate(image_names):

        print('**********')
        print(image_name)

        # Avoid duplicate names
        existed_image_num = Images.objects.filter(Q(name__startswith=image_name.split('.')[0])).count()
        if existed_image_num: # An image with a duplicate name already exists
            image_name_prefix = image_name.split('.')[0] # xxx
            image_name_postfix = image_name.split('.')[1] # png
            image_name_old = image_name
            #image_name = image_name_prefix + '(' + str(existed_image_num) + ').' + image_name_postfix # xxx(1).png
            image_name = image_name_prefix + '-' + str(existed_image_num) + '.' + image_name_postfix # xxx(1).png
        else:
            image_name_old = image_name

        # Step 1. Get thumbnail
        img_path = '%s/v0/segmentation/5x_png/%s'%(settings.BASE_DIR,image_name_old)
        image = Image.open(img_path)
        w, h = image.size
        w_scaled = 700
        h_scaled = round(w_scaled/w*h)
        image_thumbnail = image.resize((w_scaled, h_scaled), Image.ANTIALIAS)
        thumbnail_path = '%s/IMAGES/Images/%s'%(settings.BASE_DIR,image_name)
        image_thumbnail.save(thumbnail_path)
        t.refresh(track_count)
        track_count += 1

        # Step 2. Filter
        item_folder = '%s/v0/segmentation/5x_png/'%(settings.BASE_DIR)
        filtered_dir = '%s/IMAGES/filtered_png/'%(settings.BASE_DIR)
        filted_mask_dir = '%s/IMAGES/filtered_mask/'%(settings.BASE_DIR)
        apply_color_filter_to_image(image_name_old, image_name, item_folder, filtered_dir, filted_mask_dir, hole_size=2000*3000, object_size=12000)
        t.refresh(track_count)
        track_count += 1

        # Step 3. tile
        slide_list = [image_name.split('.')[0]]
        file_dir = '%s/IMAGES/filtered_png/'%(settings.BASE_DIR)
        file_mask_dir = '%s/IMAGES/filtered_mask/'%(settings.BASE_DIR)
        save_patch_dir = '%s/IMAGES/patch/'%(settings.BASE_DIR)
        sample_type = 'seg'
        patch_size = 1600
        overlap = 400
        filter_rate = 0.10
        cutter = Cutter(slide_list, file_dir, file_mask_dir, save_patch_dir, sample_type=sample_type)
        # tile info is like that {"size": [5995, 7174], "tiles": [5, 6], "step": [1098.75, 1114.8]}
        tile_info = cutter.sample_and_store_patches_png(patch_size, overlap, filter_rate)
        t.refresh(track_count)
        track_count += 1

        # Save to the database.
        original_image = Images.objects.create(
            image = '/IMAGES/Images/%s' % image_name,
            description = ' ',
            name = image_name.split('.')[0],
            postfix = image_name.split('.')[1],
            height = h_scaled,
            width = w_scaled,
            depth = 0,
            h = tile_info['size'][0],
            w = tile_info['size'][1],
            tile_vers = tile_info['tiles'][0],
            tile_cols = tile_info['tiles'][1],
            step_ver = tile_info['step'][0],
            step_col = tile_info['step'][1]
        )
        original_image.save()

        mask = Masks.objects.create(related_image=original_image, 
                                    name=image_name.split('.')[0],
                                    height = tile_info['size'][0],
                                    width = tile_info['size'][1])
        mask.save()
        # At the beginning we have no masks, so let's substitute original_image for mask.
        mask_dir = '%s/IMAGES/Masks/%s'%(settings.BASE_DIR,image_name)
        image_thumbnail.save(mask_dir)


    return HttpResponse(json.dumps({
            "status":1
            #"image_num": len(selected_images)
            }))
def process(request):
    '''
    Select images that you want to process.
    If a image has been processed, it will be skipped.
    '''
    image_list = Images.objects.order_by('-time')
    unprocessed_image_list = []
    for image in image_list:
        if image.Is_processed == False:
            unprocessed_image_list.append(image)

    template = loader.get_template('process.html')
    context = {
        'image_list': unprocessed_image_list
    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def process_handle(request):
    '''
    Process selected images, including generating masks, keypoints and diameters.
    '''
    if request.method == 'POST':
        selected_images = json.loads(request.body)
        print(selected_images)
        # Track the progress of processing data.
        track_total = 2*len(selected_images)
        track_count = 0

        # initialize segmentation network
        # Get info, like that {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}
        info = {}
        for i,selected_image in enumerate(selected_images):
            image = Images.objects.get(name=selected_image)
            slide = image.name
            size = [image.h, image.w]
            tiles = [image.tile_vers, image.tile_cols]
            step = [image.step_ver, image.step_col]
            info[slide] = {'size':size, 'tiles': tiles, 'step': step}

        seg_worker = Segmentation(config_django, selected_images, info)
        t.refresh(track_count)
        track_count += 1

        for i,selected_image in enumerate(selected_images):
            print(selected_image)
            # 1. Get mask
            seg_worker.run_segmentation(selected_image)
            t.refresh(track_count)
            track_count += 1

            # 2. Get keypoints and DOI.
            image = Images.objects.get(name=selected_image)
            mask = Masks.objects.get(related_image=image)
            mask_dir = '%s/IMAGES/Masks/%s%s'%(settings.BASE_DIR,selected_image,'.png')

            mask_keymap = KeyMap(mask_dir)
            keypoints_base = mask_keymap.search_keypoint(alpha=2) # 基准线的两个端点
            keypoints_top = mask_keymap.key_point_tumor #浸润最深处的点
            doi = mask_keymap.doi  # based on pixels
            height, width = mask_keymap.get_mask_size()

            mask.keypoint_x0 = keypoints_base[0][0]
            mask.keypoint_y0 = keypoints_base[0][1]
            mask.keypoint_x1 = keypoints_base[1][0]
            mask.keypoint_y1 = keypoints_base[1][1]
            mask.keypoint_xt = keypoints_top[0]
            mask.keypoint_yt = keypoints_top[1]
            xf, yf = get_foot_point(keypoints_base, keypoints_top)
            mask.keypoint_xf = xf
            mask.keypoint_yf = yf

            mask.width = width
            mask.height = height

            mask.save()

            scale = mask.scale
            doi = doi * scale /1000 # based on mm
            image.depth = doi
            image.Is_processed = True
            image.save()

            # 3. Get diameter of tumor.
            #mask_dir = '%s/IMAGES/Masks/%s%s'%(settings.BASE_DIR,selected_image,'.png')
            point0, point1, diameter = get_tumor_diameter(mask_dir)
            #print(diameter)


            diameter = diameter * scale /1000 # based on pixels -> based on mm
            image.diameter = diameter
            tumor_stage = judge_tumor_stage(diameter, doi) # one from 'T1' 'T2' 'T3' 'UC'
            image.tumor_stage = tumor_stage
            #image.Is_processed = True
            image.save()

            mask.dia_x0 = point0[0]
            mask.dia_y0 = point0[1]
            mask.dia_x1 = point1[0]
            mask.dia_y1 = point1[1]            
            mask.save()

            t.refresh(track_count)
            track_count += 1

        return HttpResponse(json.dumps({
            "status":selected_image
            #"image_num": len(selected_images)
        }))
    return HttpResponse('未处理')
@csrf_exempt
def process_progress(request):
    return JsonResponse(t.get_value(), safe=False)

def detail(request, name):
    '''
    Show the detail of images.
    (name with no postfix)
    '''
    # print(name)
    # print('+++++++++++++')
    image = Images.objects.get(name=name)
    mask = Masks.objects.get(related_image=image)

    objs = Images.objects.all().order_by('time')
    #next_image = objs.filter(time__gt=image.time).first()
    next_image = objs.filter(time__lt=image.time).first()

    height = mask.height
    width = mask.width
    #scaled_width = 500
    #scaled_height = round(scaled_width/width*height)

    scaled_height = 700
    scaled_width = round(scaled_height/height*width)

    keypoints = [mask.keypoint_x0*scaled_width/width,
                 mask.keypoint_y0*scaled_height/height,
                 mask.keypoint_x1*scaled_width/width,
                 mask.keypoint_y1*scaled_height/height,
                 mask.keypoint_xt*scaled_width/width,
                 mask.keypoint_yt*scaled_height/height,
                 mask.keypoint_xf*scaled_width/width,
                 mask.keypoint_yf*scaled_height/height]
    diameter_points = [mask.dia_x0*scaled_width/width,
                       mask.dia_y0*scaled_height/height,
                       mask.dia_x1*scaled_width/width,
                       mask.dia_y1*scaled_height/height,
    ]

    context = {
        'name': name,
        'next_image': next_image,
        'image': image,
        'time': image.time,
        'depth': round(image.depth, 2),
        'diameter': round(image.diameter, 2),
        'keypoints': keypoints,
        'diameter_points': diameter_points,
        'canvas_w': scaled_width,
        'canvas_h': scaled_height,
        
    }


    template = loader.get_template('detail.html')
    if request.method == 'POST':
        tumor_stage = request.POST.get('tumor_stage','')
        comment = request.POST.get('comment','')
        description = request.POST.get('description','')

        image.tumor_stage = tumor_stage
        image.comment = comment
        image.description = description
        image.save()

        context = {
        'name': name,
        'next_image': next_image,
        'image': image,
        'depth': round(image.depth, 2),
        'diameter': round(image.diameter, 2),
        'keypoints': keypoints,
        'diameter_points': diameter_points,
        'canvas_w': scaled_width,
        'canvas_h': scaled_height,
        }
        return HttpResponse(template.render(context, request))
    return HttpResponse(template.render(context, request))

def manual_keypoint(request, name):
    image = Images.objects.get(name=name)
    mask  = Masks.objects.get(related_image=image)
    height = mask.height
    width = mask.width
    scaled_height = 700
    scaled_width = round(scaled_height/height*width)
    #print(scaled_width)
    context = {
        'name': name,
        'canvas_h': scaled_height,
        'canvas_w': scaled_width,
        
    }
    template = loader.get_template('manual_keypoint.html')
    return HttpResponse(template.render(context, request))


@csrf_exempt
def manual_keypoint_handle(request):
    if request.method == 'POST':
        keypoints = json.loads(request.body)
        name = keypoints['name']
        x0 = keypoints['x0']
        y0 = keypoints['y0']
        x1 = keypoints['x1']
        y1 = keypoints['y1']
        xt = keypoints['xt']
        yt = keypoints['yt']
        xf = keypoints['xf']
        yf = keypoints['yf']
        print(x0,',',y0,',',name)

        image = Images.objects.get(name=name)
        mask = Masks.objects.get(related_image=image)
        height = mask.height
        width = mask.width
        scaled_height = 700
        scaled_width = round(scaled_height/height*width)        

        mask.keypoint_x0 = x0*width/scaled_width
        mask.keypoint_y0 = y0*height/scaled_height
        mask.keypoint_x1 = x1*width/scaled_width
        mask.keypoint_y1 = y1*height/scaled_height
        mask.keypoint_xt = xt*width/scaled_width
        mask.keypoint_yt = yt*height/scaled_height
        mask.keypoint_xf = xf*width/scaled_width
        mask.keypoint_yf = yf*height/scaled_height

        mask.save()

        scale = mask.scale
        image.depth = math.sqrt((mask.keypoint_yf-mask.keypoint_yt)**2+\
            (mask.keypoint_xf-mask.keypoint_xt)**2) * scale / 1000

        image.save()

        #return HttpResponse(keypoints)
        return HttpResponse(json.dumps({
            "status":1
        }))
    return render(request, 'manual_keypoint.html')

def process_data(request):
    for i in range(100):
        t.refresh(i)
        time.sleep(1)
    
    return JsonResponse('开始处理数据')

