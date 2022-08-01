import sys
import numpy as np
import cv2
import json
import os
from collections import OrderedDict
from argparse import ArgumentParser as ArgParse

green = (0, 255, 0)                 #이미지 내에 선이나 글자의 색상 : 녹색
red = (0, 0, 255)                   #이미지 내에 선이나 글자의 색상 : 빨강
blue = (255,0,0)                    #이미지 내에 선이나 글자의 색상 : 파랑
yellow = (0,255,255)                #이미지 내에 선이나 글자의 색상 : 노랑

NEXT_PAGE = 32                      #space key ascii code
#h_samples = [550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790 ,800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010,1020, 1030, 1040, 1050, 1060]
#h_samples = [550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790 ,800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010,1020, 1030, 1040, 1050, 1060]
h_samples = list(range(320, 720, 10))
print(h_samples)
json_file_path='./train_cart.json'  #train_cart.json 파일 경로
f = open("./train.txt", 'r')        #train image의 경로가 적힌 train.txt 파일 로드
lines = f.readlines()               #train.txt를 라인별로 읽음
file_data = OrderedDict()           #json파일 저장을 위한 file_data 선언

#글자 출력에 관한 global variable
font=cv2.FONT_HERSHEY_SIMPLEX       #폰트 종류
org = (10, 60)                      #폰트를 찍는 위치
file_count = 0                      #labeling하는 파일 차례(순서)

#lane에 대한 global variable
lane_class = 1
left_lane = 0                       #left lain은 0으로 표시함 (내가 지정했음)
right_lane = 1                      #right lain은 1로 표시함 (내가 지정했음)
lane_count = 0                      #lane의 점을 몇개 찍었는지 count

# 흰색 컬러 영상 생성
#img = np.ones((480, 640, 3), dtype=np.uint8) * 255

class MyLane:
    def __init__(self):
        self.points = []
    def points_append(self, x1, y1):
        self.points.append((x1, y1))

# def generate_label_img():    
#     label_img = np.zeros((h, w), dtype=np.uint8)
#     cv2.imshow('label_img', label_img)

def calc_inclination(lane_coordi, lane, param, h_anchor):
    #x1 : lane_coordi.points[0][0]
    #y1 : lane_coordi.points[0][1]
    #x2 : lane_coordi.points[1][0]
    #y2 : lane_coordi.points[1][1]
    m = []
    b = []            
    lane_x_axis = []
    for i in range(0, len(lane_coordi.points)-1):
        if (lane_coordi.points[i+1][0]-lane_coordi.points[i][0]) == 0 :
            print('error!!! 이전 point와 현재 point의 값이 같아서 m을 구할수 없음!!! 포인트를 찍을때 y축 기준으로 직선으로 긋지 마시오')
            os._exit(1)            
        else:
            m.append((lane_coordi.points[i+1][1]-lane_coordi.points[i][1])/(lane_coordi.points[i+1][0]-lane_coordi.points[i][0]))
            b.append(lane_coordi.points[i][1]-(m[i]*lane_coordi.points[i][0]))                         #b 는 y 절편
    print(m)
    print(b)
    count_index = 0                                                 #count index는 몇번째 anchor 인지 확인하는 flag
    print('file_data["h_samples"] : ' + str(h_anchor))
    #lane_x_axis = [int((num-b)/m) for num in h_samples]             #직선의 방정식에 따른 x좌표값 추출하는 식
    for num in h_samples:
        if num < lane_coordi.points[3][1]:
            lane_x_axis.append(-2)
        elif num < lane_coordi.points[2][1]:
            lane_x_axis.append(int((num-b[2])/m[2]))
        elif num < lane_coordi.points[1][1]:
            lane_x_axis.append(int((num-b[1])/m[1]))
        elif num < lane_coordi.points[0][1]:                                                       # 일경우가 else임
            lane_x_axis.append(int((num-b[0])/m[0]))  
        else:
            lane_x_axis.append(-2)
    print('lane_x_axis : ' + str(lane_x_axis))
    h, w, _ = param.shape                                           #이미지 사이즈의 h, w

    for i in h_samples:
        # if i >= y_top:\
        if lane_x_axis[count_index] >= 0 and lane_x_axis[count_index] <= w: #and i >= lane_coordi.points[2][1] and i <= lane_coordi.points[0][1]: #추출한 x의 좌표가 영상 안에 존재하거나, y축의 값이 anchor 값 사이일경우만 입력하고 아니면 -2 입력할것
            file_data["lanes"][lane].append(lane_x_axis[count_index])
            #print(i)
        else:
            file_data["lanes"][lane].append(-2)
        count_index += 1
    print(file_data["lanes"])
    x_axis_index = 0
    #사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함
    for i in h_samples:
        if file_data["lanes"][lane][x_axis_index] != -2:
            cv2.circle(param, (file_data["lanes"][lane][x_axis_index], i), 5, green, -1)
        x_axis_index += 1
    #cv2.line(param, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)        
    cv2.imshow('labeling_tusimple', param)
    return m, b

def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상 일수도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함
    global lane_count, left_lane_coordi, right_lane_coordi, h_samples # 밖에 있는 oldx, oldy 불러옴
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽이 눌러지면 실행        
        if lane_count == 0:            
            left_lane_coordi.points_append(x, y)                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 1:
            left_lane_coordi.points_append(x, y)                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 2:
            left_lane_coordi.points_append(x, y)                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 3:
            left_lane_coordi.points_append(x, y)                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
            calc_inclination(left_lane_coordi, left_lane, param[0], h_samples)
        elif lane_count == 4:
            right_lane_coordi.points_append(x, y)               # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 5:
            right_lane_coordi.points_append(x, y)               # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 6:
            right_lane_coordi.points_append(x, y)               # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
        elif lane_count == 7:
            right_lane_coordi.points_append(x, y)               # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
            param[1] = param[0].copy()
            calc_inclination(right_lane_coordi, right_lane, param[0], h_samples)
        else:
            pass   
        
        print('count : ' + str(lane_count))
        cv2.circle(param[0], (x, y), 5, blue, -1)
        cv2.circle(param[1], (x, y), 5, blue, -1)
        #generate_label_img()
        cv2.imshow('labeling_tusimple', param[0])
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력
        lane_count += 1

    # elif event == cv2.EVENT_LBUTTONUP: # 마우스 뗏을때 발생
    #     print('EVENT_LBUTTONUP: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스가 움직일 때 발생
        # if flags & cv2.EVENT_FLAG_LBUTTON: # ==를 쓰면 다른 키도 입력되었을 때 작동안하므로 &(and) 사용
        #     print(flags)        
        if lane_count == 0:            
            pass
        elif lane_count == 1:
            param[0] = param[1].copy()
            cv2.line(param[0], (left_lane_coordi.points[lane_count-1][0], left_lane_coordi.points[lane_count-1][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])            
        elif lane_count == 2:
            param[0] = param[1].copy()
            cv2.line(param[0], (left_lane_coordi.points[lane_count-1][0], left_lane_coordi.points[lane_count-1][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])            
        elif lane_count == 3:
            param[0] = param[1].copy()
            cv2.line(param[0], (left_lane_coordi.points[lane_count-1][0], left_lane_coordi.points[lane_count-1][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])
        elif lane_count == 4:
            pass
        elif lane_count == 5:
            param[0] = param[1].copy()
            cv2.line(param[0], (right_lane_coordi.points[lane_count-5][0], right_lane_coordi.points[lane_count-5][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])
        elif lane_count == 6:
            param[0] = param[1].copy()
            cv2.line(param[0], (right_lane_coordi.points[lane_count-5][0], right_lane_coordi.points[lane_count-5][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])
        elif lane_count == 7:
            param[0] = param[1].copy()
            cv2.line(param[0], (right_lane_coordi.points[lane_count-5][0], right_lane_coordi.points[lane_count-5][1]), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('labeling_tusimple', param[0])
        else:
            pass
            

def labeling(imagenum, left_lane_coordi, right_lane_coordi, auto_bright): 
    cv2.namedWindow('labeling_tusimple',cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty('labeling_tusimple',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #print('이까지 열림1')

    file_count = imagenum-1
    global file_data, lane_count
    while(file_count < len(lines)):
    #for line in lines: #for문으로 하니까 index 조절이 안됨
        line = lines[file_count]
        left_lane_coordi.__init__()
        right_lane_coordi.__init__()
        file_data=OrderedDict()
        lane_count = 0        
        #print('lines : ' + str(len(lines)))
        line=line.strip()
        file_data["lanes"]=[[],[]]
        file_data["h_samples"] = h_samples        
        #print('file num : ' + str(len(lines)))
        #print(line)
        img = cv2.imread(line, cv2.IMREAD_COLOR)
        img_h, img_w,_ = img.shape

        if auto_bright == 1:
            img_crop_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            yyy, cr, cb = cv2.split(img_crop_ycrcb)
            #print(ycrcb_planes.shape)

            # 밝기 성분에 대해서만 히스토그램 평활화 수행
            dst_y = cv2.equalizeHist(yyy)
            dst_ycrcb = cv2.merge([dst_y, cr, cb])
            crop = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)


        else:
            crop = img.copy()
        #crop2는 마우스 이동시 직선을 잘 그려주기 위해서 임시로 이전 frame을 저장하기 위한것
        crop2 = crop.copy()
        img_crop = [crop, crop2]

        seg_gt_png_path=(line.rstrip('.jpg ')) + '.png'        
        train_gt_str = line + ' ' + seg_gt_png_path
        label_index_txt=seg_gt_png_path.split('/')
        label_index_txt = label_index_txt[len(label_index_txt)-1].rstrip('.png')
        #label_index_txt=seg_gt_png_path.lstrip('clips/').rstrip('.png')
        train_cart_classes = ''
        #print(type(train_cart_classes))
        file_data["raw_file"] = line
        #line_clips_path = 'clips/' + file_name[len(file_name)-1]
        #file_data["raw_file"] = line_clips_path        
        #file_name=line.split('/')
        #text = str(file_count+1) + ' / ' + str(len(lines)) + ' filename : ' + file_name[len(file_name)-1]
        

        #아래 cv2.line은 사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함
        cv2.line(crop, (0, h_samples[0]), (img_w, h_samples[0]), yellow, 5, cv2.LINE_AA)             
        cv2.line(crop, (0, h_samples[len(h_samples)-1]), (img_w, h_samples[len(h_samples)-1]), yellow, 4, cv2.LINE_AA)
        #위 cv2.line은 사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함
        text = str(file_count+1) + ' / ' + str(len(lines)) + ' filename : ' + line
        cv2.putText(crop, text, org, font, 1, red, 4)    
        # 윈도우 창
        # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
        # 마우스 이벤트가 발생하면 on_mouse 함수 실행
        cv2.setMouseCallback('labeling_tusimple', on_mouse, param=img_crop)
        
        # 영상 출력
        cv2.imshow('labeling_tusimple', crop)
        waitKey=cv2.waitKey()
        #print('waitKey : ' + str(waitKey))
        if NEXT_PAGE == waitKey:
            file_count += 1                
            if lane_count >= 3:     #lane이 2개 이상 선택 되었을때 json으로 저장할것
                #print(file_data)
                #여기에 json 파일 추가해야함!!!!!!!!!!!!!                
                ##################################################cv2.imsave 해줘야함~!!!
                label_img = np.zeros((img_h, img_w), dtype=np.uint8)                
                if len(file_data["lanes"][0]) > 0:
                    lane_class = 1
                    train_gt_str = train_gt_str + ' ' + '1'
                    train_cart_classes = train_cart_classes + '1'
                    for i in range(0, len(file_data["lanes"][0])-1):
                        if file_data["lanes"][0][i] != -2 and file_data["lanes"][0][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][0][i], h_samples[i]), (file_data["lanes"][0][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)
                else:
                    train_gt_str = train_gt_str + ' ' + '0'
                    train_cart_classes = train_cart_classes + '0'
                if len(file_data["lanes"][1]) > 0:
                    train_gt_str = train_gt_str + ' ' + '1'
                    train_cart_classes = train_cart_classes + ' 1'
                    lane_class = 2
                    # if train_cart_classes == '1':
                    #     lane_class = 2
                    #     train_cart_classes = train_cart_classes + ' 1'
                    # else:
                    #     train_cart_classes = train_cart_classes + '1'
                    #     lane_class = 1
                    for i in range(0, len(file_data["lanes"][1])-1):
                        if file_data["lanes"][1][i] != -2 and file_data["lanes"][1][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][1][i], h_samples[i]), (file_data["lanes"][1][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)
                else:
                    train_gt_str = train_gt_str + ' ' + '0'
                    train_cart_classes = train_cart_classes + ' 0'
                train_gt_str = train_gt_str + '\n'                
                if train_cart_classes != None:
                    train_cart_classes=train_cart_classes+'\n'                    

                with open("./train_cart_classes.txt", 'r+') as gt_classes_txt:
                    for line in gt_classes_txt:
                        pass
                    gt_classes_txt.write(train_cart_classes)

                with open("./train_gt.txt", 'r+') as gt_txt:
                    for line in gt_txt:
                        pass
                    gt_txt.write(train_gt_str)
                #gt_lines = gt_txt.readlines()
                #cv2.imshow('label_img', label_img)     #seg label 보고싶으면 이걸 활성화 할것
                cv2.imwrite(seg_gt_png_path, label_img)
                ##################################################cv2.imsave 해줘야함~!!!
                with open(json_file_path, "r+") as json_file:
                    for line in json_file:                  #파일의 맨 끝으로 가는 코드                        
                        pass
                    string = json.dumps(file_data)
                    string += '\n'
                    json_file.write(string)                
                print(label_index_txt)
                with open("./label_index.txt", 'w') as label_index_str:
                    label_index_str.write(label_index_txt)
            continue
        elif waitKey == 3 or waitKey == 54:      #-> 방향키 눌렀을때 다음 이미지로 그냥 넘어가고 라벨링은 안됨
            file_count += 1
        elif waitKey == 2 or waitKey == 52:      #<- 방향키 눌렀을때 이전 이미지로 그냥 넘어가고 라벨링은 안됨
            if(file_count > 0):
                file_count -= 1
            else:
                file_count = 0
        elif waitKey == 49:      #키보드 1을 누르면 auto_bright가 바뀜
            if auto_bright == 1:
                auto_bright = 0
                print('auto_bright off')
            else:
                auto_bright = 1
                print('auto_bright on')

        elif waitKey== ord('q') or waitKey==ord('Q') or waitKey==66: #'q' (113) 나 'Q' (81) 누르면 for문에서 빠져나가도록
            #json 저장 함수 호출 작성해야함!!!!!!!!!!!
            break
    cv2.destroyAllWindows()
    f.close()

if __name__=='__main__':
    ap = ArgParse()
    ap.add_argument('--imagenum', type=int, default=0)
    ap.add_argument('--auto_bright', type=int, default=0)
    #ap.add_argument('--labels', type=str, default='label_data_0313.json')

    args = ap.parse_args()

    #label_index.txt 파일에 적힌 번호 다음 번호를 받아와서 이미지를 로드함
    label_index = 0
    with open("./label_index.txt", 'r') as label_index_num:
        readline_label_index=label_index_num.readlines()            
        if readline_label_index != None:                        
            label_index=int(readline_label_index[0])+1
    if label_index != 0 and args.imagenum == 0:
        args.imagenum = label_index
        print('label_index.txt 파일에 의해 ' + str(label_index) + '번 이미지가 열림')        
    #--imagenum argument로 사용자에게 이미지 번호를 받아서 로드함
    elif args.imagenum !=0:
        print('사용자에 의해 ' + str(args.imagenum) + '번 이미지가 열림')        


    left_lane_coordi = MyLane()
    right_lane_coordi = MyLane()

    labeling(args.imagenum, left_lane_coordi, right_lane_coordi, args.auto_bright)