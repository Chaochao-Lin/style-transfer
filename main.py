import tensorflow as tf
import numpy as np
import typing
import os
from tqdm import tqdm

###config start###
CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5} #内容特征层及loss加权系数
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}                        #风格特征层及loss加权系数

CONTENT_IMAGE_PATH = './data/content.jpg'          #内容图片路径
STYLE_IMAGE_PATH = './data/style.jpg'              #风格图片路径
OUTPUT_DIR = './output'                            #生成图片的保存目录

CONTENT_LOSS_FACTOR = 1      #内容loss总加权系数
STYLE_LOSS_FACTOR = 100      #风格loss总加权系数

WIDTH = 450                  #图片宽度
HEIGHT = 300                 #图片高度

EPOCHS = 20                  #训练epoch数
STEPS_PER_EPOCH = 100        #每个epoch训练多少次
LEARNING_RATE = 0.03         #学习率
### config end ###

#创建保存生成图片的文件夹
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

###func of img_process start###
image_mean = tf.constant([0.485, 0.456, 0.406])      #经典网络img平均值
image_std = tf.constant([0.299, 0.224, 0.225])       #经典网络img标准差


#图片归一化
def normalization(image):
    return (image - image_mean) / image_std


#加载并处理图片
def load_image(image_path, width=WIDTH, height=HEIGHT):
    img = tf.io.read_file(image_path)                    #加载文件
    img = tf.image.decode_jpeg(img, channels=3)          #解码图片，彩色图片cannel=3
    img = tf.image.resize(img, [height, width],  method=tf.image.ResizeMethod.BILINEAR)          #修改图片大小
    img = img / 255.
    img = normalization(img)                             #归一化
    img = tf.reshape(img, [1, height, width, 3])
    return img


#保存图片
def save_image(image, filename):
    img = tf.reshape(image, image.shape[1:])#height,width,channels
    img = img * image_std + image_mean      #反归一化
    img = img * 255.                        #修改图片大小
    img = tf.cast(img, tf.int32)            #数据类型转换，32位整型
    img = tf.clip_by_value(img, 0, 255)     #将张量数值限定在0-255即rgb
    img = tf.cast(img, tf.uint8)            #数据类型转换，8位无符号整型
    img = tf.image.encode_jpeg(img)         #编码图片，彩色图片cannel=3
    tf.io.write_file(filename, img)         #写入图片文件
### func of img_process end ###


###model setting start###
#创建并初始化vgg19模型
def get_vgg19_model(layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')  #加载imagenet上预训练的vgg19
    outputs = [vgg.get_layer(layer).output for layer in layers]  #提取需要被用到的vgg的层到outputs
    model = tf.keras.Model([vgg.input, ], outputs)               #使用outputs创建新的模型
    model.trainable = False                                      #锁死参数，不进行训练
    return model


class NeuralStyleTransferModel(tf.keras.Model):
    def __init__(self, content_layers: typing.Dict[str, float] = CONTENT_LAYERS,
                 style_layers: typing.Dict[str, float] = STYLE_LAYERS):
        super(NeuralStyleTransferModel, self).__init__()#继承父类初始化
        self.content_layers = content_layers            #内容特征层字典 Dict[层名,加权系数]
        self.style_layers = style_layers                #风格特征层
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())  #提取需要用到的所有vgg层
        self.outputs_index_map = dict(zip(layers, range(len(layers))))     #创建layer_name到output索引的映射
        self.vgg = get_vgg19_model(layers)              #创建并初始化vgg网络

    def call(self, inputs, ):
        outputs = self.vgg(inputs)
        content_outputs = []     #分离内容特征层和风格training=None, mask=None特征层的输出，方便后续计算 typing.List[outputs,加权系数]
        for layer, factor in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        # 以字典的形式返回输出typing.Dict[str,typing.List[outputs,加权系数]]
        return {'content': content_outputs, 'style': style_outputs}
### model setting end ###


###train start###
model = NeuralStyleTransferModel()                  #创建模型
content_image = load_image(CONTENT_IMAGE_PATH)     #加载内容图片
style_image = load_image(STYLE_IMAGE_PATH)  #加载风格图片
target_content_features = model([content_image, ])['content']   #计算出目标内容图片的内容特征备用
target_style_features = model([style_image, ])['style']         #计算目标风格图片的风格特征
M = WIDTH * HEIGHT
N = 3


#计算指定层上两个特征之间的内容loss
def _compute_content_loss(noise_features, target_features):
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    x = 2. * M * N             #计算系数
    return content_loss / x


#计算并减小当前图片的内容loss
def compute_content_loss(noise_content_features):
    content_losses = []         #初始化内容损失
    #加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


#计算给定特征的Gram矩阵
def gram_matrix(feature):
    x = tf.transpose(feature, perm=[2, 0, 1])   #先交换维度，把channel维度提到最前面
    x = tf.reshape(x, (x.shape[0], -1))         #reshape，压缩成2d，（channels，height*width）
    return x @ tf.transpose(x)                  #矩阵乘法计算Gram矩阵


#计算指定层上两个特征之间的风格loss
def _compute_style_loss(noise_feature, target_feature):
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    x = 4. * (M ** 2) * (N ** 2)                #计算系数
    return style_loss / x


#计算并返回图片的风格loss
def compute_style_loss(noise_style_features):
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)


#计算总损失
def total_loss(noise_features):
    content_loss = compute_content_loss(noise_features['content'])
    style_loss = compute_style_loss(noise_features['style'])
    return content_loss * CONTENT_LOSS_FACTOR + style_loss * STYLE_LOSS_FACTOR


optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)    #使用Adma优化器
#基于内容图片随机生成一张噪声图片
noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, HEIGHT, WIDTH, 3))) / 2)

#一次迭代过程
@tf.function        #使用tf.function加速训练
def train_one_step():
    #求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    grad = tape.gradient(loss, noise_image)             #求梯度
    optimizer.apply_gradients([(grad, noise_image)])    #梯度下降，更新噪声图片
    return loss


#共训练EPOCHS个epochs
for epoch in range(EPOCHS):
    #使用tqdm库提示训练进度
    with tqdm(total=STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, EPOCHS)) as pbar:
        #每个epoch训练STEPS_PER_EPOCH次
        for step in range(STEPS_PER_EPOCH):
            _loss = train_one_step()
            pbar.set_postfix({'loss': '%.4f' % float(_loss)})
            pbar.update(1)
        #每个epoch保存一次图片
        save_image(noise_image, '{}/{}.jpg'.format(OUTPUT_DIR, epoch + 1))
### train end ###
