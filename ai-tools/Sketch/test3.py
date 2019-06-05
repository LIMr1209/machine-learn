from PIL import Image
#image 是 PIL库中代表一个图像的类
import numpy as np

#打开一张图片 是图片位置
a = np.asarray(Image.open('test.jpg')
               .convert('L')).astype('float')


depth = 10.                    #浮点数，预设深度值为10
grad = np.gradient(a)          #取图像灰度的梯度值
grad_x,grad_y = grad           #分别取横纵图像的梯度值
grad_x = grad_x*depth/100.     #根据深度调整 x 和 y 方向的梯度值
grad_y = grad_y*depth/100.
A = np.sqrt(grad_x**2 + grad_y**2 +1.)      #构造x和y轴梯度的三维归一化单位坐标系
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A

vec_el = np.pi/2.2                       #光源的俯视角度，弧度值
vec_az = np.pi/4.                        #光源的方位角度，弧度值
dx = np.cos(vec_el)*np.cos(vec_az)       #光源对 x 轴的影响，np.cos(vec_el)为单位光线在地平面上的投影长度
dy = np.cos(vec_el)*np.sin(vec_az)       #光源对 y 轴的影响
dz = np.sin(vec_el)                      #光源对 z 轴的影响

b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)    #梯度与光源相互作用，将梯度转化为灰度
b = b.clip(0,255)                          #为避免数据越界，将生成的灰度值裁剪至0‐255区间

im = Image.fromarray(b.astype('uint8'))     #重构图像
im.save("test3.jpg")      #保存图片的地址
