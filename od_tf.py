import cv2 as cv
import matplotlib.pyplot as plt

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv.dnn_DetectionModel(frozen_model,config_file)

classLabels = []

file_name = 'Labels.txt'
with open(file_name,'rt') as fpt :
    classLabels = fpt.read().rstrip('\n').split('\n')


img = cv.imread('wp2359348-bmw-drift-wallpapers.jpg')
plt.imshow(img)
# plt.show()
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

ClassIndex, confidece, bbox = model.detect(img,confThreshold = 0.5)
font_scale = 3
font = cv.FONT_HERSHEY_PLAIN
for ClassInd, conf , boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
    cv.rectangle(img,boxes,(255,0,0),2)
    cv.putText(img,classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40),font,fontScale = font_scale,color=(0,255,0),thickness=3)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.show()



cap = cv.VideoCapture(0)
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open the video")

while True:
    ret,frame = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame,confThreshold = 0.55)
    print(classLabels[ClassInd-1])
    if(len(ClassIndex)!=0):
        for ClassInd, conf , boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv.rectangle(frame,boxes,(255,0,0),2)
                cv.putText(frame,classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40),font,fontScale = font_scale,color=(0,255,0),thickness=3)
    cv.imshow("ob_dete",frame)

    if cv.waitKey(2) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows    
