import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
img_path = "test_images/face_extraction/test6.jpeg"
img_name = img_path.split(".")[0]
image = cv2.imread(img_path)

temp = image.copy()
result = detector.detect_faces(image)
if(len(result)==0):
	print("No face detect. Invalid/low-quality face image")
	exit()

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(temp,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

x, y, w, h = result[0]['box']

# print(x,y,w,h)
# width,height = image.size[:2]

face = image[y-20:y+h+20, x-20:x+w+20]
cv2.imshow("temp", face)
cv2.waitKey(0);
cv2.imwrite(img_name + "_face.jpg", face)

# print(result)