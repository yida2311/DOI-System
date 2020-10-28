import json, datetime

class tracker():
    def __init__(self):
        self.value = None
    def refresh(self,v):
        self.value = v
    def get_value(self):
        return self.value
    def __str__(self):
        return self.value

def get_foot_point(keypoints_base, keypoints_top):
    '''
    获得垂足，x0,y0,x1,y1确定直线，过xt,yt向该直线作垂线，求得垂足坐标
    '''
    x0 = keypoints_base[0][0]
    y0 = keypoints_base[0][1]
    x1 = keypoints_base[1][0]
    y1 = keypoints_base[1][1]   
    xt = keypoints_top[0]
    yt = keypoints_top[1]

    k = -((x0-xt)*(x1-x0) + (y0-yt)*(y1-y0)) / ((x1 - x0)**2 + (y1 - y0)**2)*1.0
    xf = k*(x1-x0)+x0
    yf = k*(y1-y0)+y0

    return xf, yf

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self,obj)

def cut_string_length(whole_string, cut_length):
    if len(whole_string) <= cut_length:
        return whole_string
    else:
        return whole_string[0: cut_length] + '...'

def judge_tumor_stage(diameter, doi):
    '''
    T1: diameter<=2cm and doi<=5mm
    T2: (diameter<=2cm and 5mm<doi<=10mm) OR (2cm<diameter<=4cm and doi<=10mm)
    T3: diameter>4cm OR doi>10mm
    '''
    if diameter>40 or doi>10:
        return 'T3'
    if diameter<=20 and doi<=5:
        return 'T1'
    if diameter<=20 and (doi>5 and doi<=10):
        return 'T2'
    if (diameter>20 and diameter<=40) and doi<=10:
        return 'T2'
    return 'UC'



if __name__ == '__main__':
    print(judge_tumor_stage(10, 2)) # T1
    print(judge_tumor_stage(10, 6)) # T2
    print(judge_tumor_stage(30, 2)) # T2
    print(judge_tumor_stage(50, 2)) # T3
    print(judge_tumor_stage(4, 12)) # T3
    print(judge_tumor_stage(1, 2)) # T1