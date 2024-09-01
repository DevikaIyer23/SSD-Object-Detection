# SSD-Object-Detection-
The objective of this project is to robustly analyse an image and accurately identify the two prevalent solid state drivers namely the SSD 2230 and SSD 2280 using a yolo v8 model. It aims to automate computer hardware identification within images thereby distinguishing the hard disks based on key physical characteristics


##Data preprocessing
1. Consolidating the Dataset with equal images of the SSD2230 and SSD2280 variants of the hard disk. Having equal number of images of both classes ensures minimal bias.
2. Annotating the images using LabelImg. The object is bound by a rectangular box to denote which region contains the object and annotated with a label (SSD2230 or SSD2280). The image is then stored in an xml format.
3. Split the dataset into training and evaluation datasets.

##Model training and evaluation
The yolo v8 model is trained on the above custom dataset. The evaluation metrics are:
mAP=91.2%
accuracy=91.4%
precision=89.3%
recall=75%


