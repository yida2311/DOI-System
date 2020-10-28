# -*- coding: utf-8 -*-

import math

#获取基准点的下标
def get_leftbottompoint(p):
    k = 0
    for i in range(1, len(p)):
        if p[i]['y'] < p[k]['y'] or (p[i]['y'] == p[k]['y'] and p[i]['x'] < p[k]['x']):
            k = i
    return k

#叉乘计算方法
def multiply(p1, p2, p0):
    return (p1['x'] - p0['x']) * (p2['y'] - p0['y']) - (p2['x'] - p0['x']) * (p1['y'] - p0['y'])

#获取极角，通过求反正切得出，考虑pi / 2的情况
def get_arc(p1, p0):
    # 兼容sort_points_tan的考虑
    if (p1['x'] - p0['x']) == 0:

        if ((p1['y'] - p0['y'])) == 0:
            return -1;
        else:
            return math.pi / 2

    tan = float((p1['y'] - p0['y'])) / float((p1['x'] - p0['x']))
    arc = math.atan(tan)
    if arc >= 0:
        return arc
    else:
        return math.pi + arc

#对极角进行排序
def sort_points_tan(p, k):
    p2 = []
    for i in range(0, len(p)):
        p2.append({"index": i, "arc": get_arc(p[i], p[k])})
    p2.sort(key=lambda k: (k.get('arc', 0)))
    p_out = []
    for i in range(0, len(p2)):
        p_out.append(p[p2[i]["index"]])
    return p_out


def graham_scan(p):
    k = get_leftbottompoint(p)
    p_sort = sort_points_tan(p, k)

    p_result = [None] * len(p_sort)
    p_result[0] = p_sort[0]
    p_result[1] = p_sort[1]
    p_result[2] = p_sort[2]

    top = 2
    for i in range(3, len(p_sort)):
        #叉乘为正则符合条件
        while (top >= 1 and multiply(p_sort[i], p_result[top], p_result[top - 1]) > 0):
            top -= 1
        top += 1
        p_result[top] = p_sort[i]

    for i in range(len(p_result) - 1, -1, -1):
        if p_result[i] == None:
            p_result.pop()

    return p_result


#测试
ps = [{"x": 2, "y": 2}, {"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 1.5, "y": 1.5}, {"x": 1, "y": 2}, {"x": 3, "y": 1.5},
      {"x": 1.5, "y": 1.2}, {"x": 0.5, "y": 2}, {"x": 1.5, "y": 0.5}]

print(graham_scan(ps))