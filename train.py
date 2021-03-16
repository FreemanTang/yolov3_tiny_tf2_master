# 导入相关的库
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
# 导入自定义的库
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

# 设置命令行传参的关键字
flags.DEFINE_string('dataset', '', 'path to dataset') # 训练数据集
flags.DEFINE_string('val_dataset', '', 'path to validation dataset') # 验证数据集
flags.DEFINE_boolean('tiny', True, 'yolov3-tiny') # 是否训练tiny模型

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file') # 类别文件

flags.DEFINE_integer('size', 416, 'image size') # 输入图片大小
flags.DEFINE_integer('epochs', 2, 'number of epochs') # 周期数
flags.DEFINE_integer('batch_size', 8, 'batch size') # 批次
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate') # 学习率
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model') # 类别数

# 定义主函数
def main(_argv):
    # 使用GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    
    if FLAGS.tiny: # tiny模型
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors # 锚框
        anchor_masks = yolo_tiny_anchor_masks # 锚框对应的索引值

    if FLAGS.dataset: # 读取训练数据集
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    # 用来打乱数据集中数据顺序，训练时非常常用,取所有数据的前buffer_size数据项
    train_dataset = train_dataset.shuffle(buffer_size=512)
    # 设置批次,按照顺序取出batch_size行数据，最后一次输出可能小于batch
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    # 训练数据,格式(x,y)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    # 预先载入buffer_size项
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:# 读取验证数据集
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    # 设置批次,按照顺序取出batch_size行数据，最后一次输出可能小于batch
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    # 验证数据,格式(x,y)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

            
    # 优化器
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    # 损失函数
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    # 编译模型
    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=False)
    # 回调函数列表
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_tiny_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]
    # 训练模型
    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main) # 运行主函数
    except SystemExit:
        pass
