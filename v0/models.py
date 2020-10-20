from django.db import models

image_height = 400
image_width = 400

class Images(models.Model):
    '''
    The scaled and filtered images, obtained from *.svs files.
    '''
    image = models.ImageField(upload_to='Images/')
    description = models.CharField(max_length=50, blank=True) # The description of image
    name = models.CharField(max_length=30)
    postfix = models.CharField(max_length=5, default='png') # png、jpg...
    time = models.DateTimeField(auto_now_add=True)
    height = models.IntegerField() # Here height and width correspond to the thumbnail image.
    width = models.IntegerField()

    depth = models.FloatField(default=0.0) # 浸润深度
    diameter = models.FloatField(default=0.0) # 肿瘤直径
    StageOfTumour = [   
        ('UC','Unconfirmed'), 
        ('T1','Tier1'),
        ('T2','Tier2'),
        ('T3','Tier3'),
        ('T4A','Tier4A'),
        ('T4B','Tier4B'),
    ]
    tumor_stage = models.CharField(max_length=3, choices=StageOfTumour, default='UC',)

    Comment = [
        ('UC','Unconfirmed'),
        ('AP','Approved'),
        ('UA','Unaccepted'),
    ]
    comment = models.CharField(max_length=2, choices=Comment, default='UC',)

    Is_processed = models.BooleanField(default=False) # 是否处理过得到mask和关键点

    # Tile info
    # Tile info was initially stored as .json, here we store it in the database.
    # The json file is like that {"20190718213917": {"size": [5995, 7174], "tiles": [5, 6], "step": [1098.75, 1114.8]}
    h = models.IntegerField(default=6000) # Here h and w correspond to the scale down 8x image.
    w = models.IntegerField(default=7000)
    tile_vers = models.IntegerField(default=5)
    tile_cols = models.IntegerField(default=6)
    step_ver = models.FloatField(default=1000)
    step_col = models.FloatField(default=1000)




    def __str__(self):
        return self.name


class Masks(models.Model):
    '''
    The segmentation masks of images.
    '''
    name = models.CharField(max_length=30)
    related_image = models.OneToOneField(Images, on_delete=models.CASCADE)

    keypoint_x0 = models.IntegerField(default=0) # 基准线的一个点
    keypoint_y0 = models.IntegerField(default=0)
    keypoint_x1 = models.IntegerField(default=1) # 基准线的另一个点
    keypoint_y1 = models.IntegerField(default=1)
    keypoint_xt = models.IntegerField(default=2) # 浸入最深处的点
    keypoint_yt = models.IntegerField(default=2)
    keypoint_xf = models.IntegerField(default=3) # 过浸入最深处点向基准线作垂线，交于垂足(xf,yf)
    keypoint_yf = models.IntegerField(default=3)

    dia_x0 = models.IntegerField(default=0) # 肿瘤直径的两个端点
    dia_y0 = models.IntegerField(default=0)
    dia_x1 = models.IntegerField(default=0)
    dia_y1 = models.IntegerField(default=0)

    height = models.IntegerField(default=5000)
    width = models.IntegerField(default=3000)    

    scale = models.FloatField(default=0.261324*8) # 一个像素对应0.26*8微米
    manual_keypoint = models.BooleanField(default=False) # 是否人工标注过关键点
    def __str__(self):
        return self.name




# class Keypoint_masks(models.Model):
#     '''
#     Masks with keypoints.
#     '''
#     name = models.CharField(max_length=30)
#     related_image = models.OneToOneField(Images, on_delete=models.CASCADE)



#     manual_keypoint = models.BooleanField(default=False)

#     invasion_depth = models.IntegerField(default=0)