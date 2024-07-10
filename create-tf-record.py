import tensorflow as tf
from object_detection.utils import dataset_util
import os
import io
import PIL.Image
from lxml import etree

def create_tf_example(example):
    img_path = example['img_path']
    xml_path = example['xml_path']

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    image = PIL.Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()

    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    filename = data['filename'].encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin']) / width
        xmax = float(obj['bndbox']['xmax']) / width
        ymin = float(obj['bndbox']['ymin']) / height
        ymax = float(obj['bndbox']['ymax']) / height

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(1 if obj['name'] == 'SSD2230' else 2)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(output_path, examples):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Creating TFRecord file at {output_path}")

    writer = tf.io.TFRecordWriter(output_path)
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f"TFRecord file created at {output_path}")

train_examples = [
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 1.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 1.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 2.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 2.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 3.jpeg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 3.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 4.jpeg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 4.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 5.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 5.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 6.jpeg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 6.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 7.jpeg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 7.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 11.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 11.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 12.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 12.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 13.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 13.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 14.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 14.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 15.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 15.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 16.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 16.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 17.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 17.xml'}
]

eval_examples = [
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 8.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 8.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 9.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 9.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 10.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 10.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 18.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 18.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 19.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 19.xml'},
    {'img_path': r'C:/Users/devik/OneDrive/Desktop/dataset/images/img 20.jpg', 'xml_path': r'C:/Users/devik/OneDrive/Desktop/dataset/annotations/img 20.xml'}
]

create_tf_record('C:/Users/devik/OneDrive/Desktop/dataset/train/output.tfrecord', train_examples)
create_tf_record('C:/Users/devik/OneDrive/Desktop/dataset/eval/output.tfrecord', eval_examples)
