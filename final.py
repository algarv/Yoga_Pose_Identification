import cv2
import mediapipe as mp
from numpy.core.fromnumeric import nonzero 
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

video_steam = cv2.VideoCapture(0)

pTime = 0
x = 0

def find_angle(pt1,pt2,pt3):
    vector_1 = [pt1[0] - pt2[0], pt1[1] - pt2[1]]  
    vector_2 = [pt2[0] - pt3[0], pt2[1] - pt3[1]]
    mag1 = np.sqrt(vector_1[0]**2 + vector_1[1]**2)
    normed1 = vector_1 / mag1
    mag2 = np.sqrt(vector_2[0]**2 + vector_2[1]**2)
    normed2 = vector_2 / mag2
    dot = normed1[0]*normed2[0] + normed1[1]*normed2[1]
    angle_rad = np.arccos(dot)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg

tree_img = cv2.imread('tree.png')
star_img = cv2.imread('star2.png')
L_chair_img = cv2.flip(cv2.imread('chair.png'),1)
R_chair_img = cv2.imread('chair.png')

while True:
    success, original_img = video_steam.read()
    #original_img = cv2.imread('tree.jpg')

    original_r = original_img.shape[0]
    original_c = original_img.shape[1]

    img = cv2.resize(original_img,[int(.6*original_c),int(.6*original_r)])

    r = img.shape[0]
    c = img.shape[1]

    RGB_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = pose.process(RGB_img)
    landmark_list = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            landmark_list.append([int(lm.x*c), int(lm.y*r)])
            cx, cy = int(lm.x*c), int(lm.y*r)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        
        nose = landmark_list[0]
        left_shoulder = landmark_list[11]
        right_shoulder = landmark_list[12]
        left_elbow = landmark_list[13]
        right_elbow = landmark_list[14]
        left_wrist = landmark_list[15]
        right_wrist = landmark_list[16]
        left_hip = landmark_list[23]
        right_hip = landmark_list[24]
        left_knee = landmark_list[25]    
        right_knee = landmark_list[26]
        left_ankle = landmark_list[27]
        right_ankle = landmark_list[28]

        height = np.maximum(abs(right_ankle[1] - nose[1]),abs(left_ankle[1] - nose[1]))

        width = max([abs(left_wrist[0] - right_wrist[0]),abs(left_elbow[0] - right_elbow[0]),abs(left_wrist[0]-left_hip[0])])

        hip_knee_ankleL = find_angle(left_hip,left_knee,left_ankle)
        hip_knee_ankleR = find_angle(right_hip,right_knee,right_ankle)
        shoulder_elbow_wristL = find_angle(left_shoulder,left_elbow,left_wrist)
        shoulder_elbow_wristR = find_angle(right_shoulder,right_elbow,right_wrist)
        hip_shoulder_wristR = find_angle(right_hip,right_shoulder,right_wrist)
        elbow_shoulder_shoulderL = find_angle(left_elbow, left_shoulder, right_shoulder)
        elbow_shoulder_shoulderR = find_angle(right_elbow, right_shoulder, left_shoulder)

        output = img.copy()

        if (hip_knee_ankleL < 10 and hip_knee_ankleR > 90) or (hip_knee_ankleR < 10 and hip_knee_ankleL > 90):
            detected_pose = 'tree'
            pose_img = tree_img 
        elif shoulder_elbow_wristL < 20 and shoulder_elbow_wristR < 20 and elbow_shoulder_shoulderL < 20 and elbow_shoulder_shoulderR < 20:
            detected_pose = 'star'
            pose_img = star_img
        elif (80 < hip_knee_ankleR and hip_knee_ankleR < 120) or (80 < hip_knee_ankleR and hip_knee_ankleR < 120):
            if (nose[0] - left_hip[0]) < 0:
                detected_pose = 'L_chair'
                pose_img = L_chair_img
            else:
                detected_pose = 'R_chair'
                pose_img = R_chair_img
        else:
            detected_pose = None

        print(detected_pose)

        if detected_pose:
            pose_img = cv2.resize(pose_img,[width,height],cv2.INTER_LINEAR)

            h = pose_img.shape[0]
            w = pose_img.shape[1]

            # cv2.imshow('Original',pose_img)
            # cv2.waitKey(1)

            a = int(abs(h-r)/2)
            b = int(abs(w-c)/2)
            if detected_pose == 'L_chair' or detected_pose == 'R_chair':
                template_img = cv2.copyMakeBorder(pose_img, 2*a,0, b, b, cv2.BORDER_CONSTANT, None, (255,255,255))
            else:
                template_img = cv2.copyMakeBorder(pose_img, a, a, b, b, cv2.BORDER_CONSTANT, None, (255,255,255))
            for i in range(template_img.shape[0]):
                for j in range(template_img.shape[1]):
                    if template_img[i][j][0] < 50 and template_img[i][j][1] < 50 and template_img[i][j][2] < 50:
                        template_img[i][j] = [255,255,255]
        
            template_img_bw = cv2.cvtColor(template_img.copy(),cv2.COLOR_BGR2GRAY)          

            if detected_pose == 'star':
                
                pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = cv2.goodFeaturesToTrack(template_img_bw,10,.005,.75)
                pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]

                sorted_byrows = np.sort([pt1[1], pt2[1], pt3[1], pt4[1], pt5[1], pt6[1], pt7[1], pt8[1], pt9[1], pt10[1]])
                sorted_bycolumns = np.sort([pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]])

                for pt in [pt1,pt2,pt3,pt4,pt5,pt5,pt6,pt7,pt8,pt9,pt10]:
                    if pt[1] == sorted_byrows[0]:
                        head_match = [pt[1], pt[0]]
                        #print('Found head: ',head_match)
                    elif pt[1] >= sorted_byrows[8]:
                        if pt[0] <= sorted_bycolumns[2]:
                            right_ankle_match = [pt[1], pt[0]]
                            #print('Found right ankle: ', right_ankle_match)
                        else:
                            left_ankle_match = [pt[1], pt[0]]
                            #print('Found left ankle: ', left_ankle_match)
                    elif pt[0] == sorted_bycolumns[0]:
                        right_wrist_match = [pt[1], pt[0]]
                        #print('Found left wrist: ', right_wrist_match)
                    elif pt[0] == sorted_bycolumns[9]:
                        left_wrist_match = [pt[1], pt[0]]
                        #print('Found right wrist: ', left_wrist_match)
                try:
                    matched_points = np.array([left_ankle_match, right_ankle_match, head_match]).astype(np.float32)
                    key_bodypts = np.array([[left_ankle[1],left_ankle[0]], [right_ankle[1],right_ankle[0]], [nose[1],nose[0]]]).astype(np.float32)
                except:
                    print('Landmarks Not Found')
                    cv2.imshow("Image", output)
                    cv2.waitKey(1)
                    continue

            if detected_pose == 'tree':

                bool_img = template_img_bw != 255
                nonzero_columns, nonzero_rows = np.nonzero(bool_img)

                nonzero_columns = nonzero_columns.tolist()
                nonzero_rows = nonzero_rows.tolist()

                head_index = nonzero_columns.index(min(nonzero_columns))
                head_match = [nonzero_columns[head_index],nonzero_rows[head_index]]

                shoulder_index = nonzero_rows.index(min(nonzero_rows))
                right_shoulder_match = [nonzero_columns[shoulder_index],nonzero_rows[shoulder_index]]
                
                ankle_index = nonzero_columns.index(max(nonzero_columns))
                left_ankle_match = [nonzero_columns[ankle_index],nonzero_rows[ankle_index]]

                try: 
                    matched_points = np.array([left_ankle_match, right_shoulder_match, head_match]).astype(np.float32)
                    key_bodypts = np.array([[left_ankle[1],left_ankle[0]-20], [right_elbow[1],right_elbow[0]-20], [nose[1],nose[0]]]).astype(np.float32)
                except:
                    print('Landmarks Not Found')
                    cv2.imshow("Image", output)
                    cv2.waitKey(1)
                    continue

            if detected_pose == 'L_chair':
                chair_dilated = cv2.dilate(template_img_bw,np.ones([5,5]))
                chair_edges = cv2.Canny(chair_dilated,250,255)

                try:
                    pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = cv2.goodFeaturesToTrack(template_img_bw,10,.5,.75)
                    pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]
                except:
                    continue

                sorted_byrows = np.sort([pt1[1], pt2[1], pt3[1], pt4[1], pt5[1], pt6[1], pt7[1], pt8[1], pt9[1], pt10[1]])
                sorted_bycolumns = np.sort([pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]])

                for pt in [pt1,pt2,pt3,pt4,pt5,pt5,pt6,pt7,pt8,pt9,pt10]:
                    if pt[1] >= sorted_byrows[8]:
                        if pt[0] <= sorted_bycolumns[5]:
                            left_ankle_match = [pt[1]+20, pt[0]]
                            #print('Found left ankle: ',left_ankle_match)
                    elif pt[0] == sorted_bycolumns[0]:
                        left_knee_match = [pt[1]-10, pt[0]-20]
                        #print('Found left knee: ', left_knee_match)
                    elif pt[0] >= sorted_bycolumns[5]:
                        if pt[1] <= sorted_byrows[3]:
                            left_hip_match = [pt[1]-30, pt[0]]
                            #print('Found left hip: ', left_hip_match)
                try:
                    matched_points = np.array([left_hip_match,left_knee_match,left_ankle_match]).astype(np.float32)
                    key_bodypts = np.array([[left_hip[1],left_hip[0]],[left_knee[1],left_knee[0]],[left_ankle[1],left_ankle[0]]]).astype(np.float32)
                except:
                    print('Landmarks Not Found')
                    cv2.imshow("Image", output)
                    cv2.waitKey(1)
                    continue

            if detected_pose == 'R_chair':
                chair_dilated = cv2.dilate(template_img_bw,np.ones([5,5]))
                chair_edges = cv2.Canny(chair_dilated,250,255)

                # try:
                #     pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = cv2.goodFeaturesToTrack(template_img_bw,10,.5,.75)
                #     pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10 = pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]
                # except:
                #     continue

                # sorted_byrows = np.sort([pt1[1], pt2[1], pt3[1], pt4[1], pt5[1], pt6[1], pt7[1], pt8[1], pt9[1], pt10[1]])
                # sorted_bycolumns = np.sort([pt1[0], pt2[0], pt3[0], pt4[0], pt5[0], pt6[0], pt7[0], pt8[0], pt9[0], pt10[0]])

                # for pt in [pt1,pt2,pt3,pt4,pt5,pt5,pt6,pt7,pt8,pt9,pt10]:
                #     if pt[1] >= sorted_byrows[8]:
                #         if pt[0] >= sorted_bycolumns[4]:
                #             right_ankle_match = [pt[1]-20, pt[0]]
                #             # print('Found left ankle: ',left_ankle_match)
                #     elif pt[1] <= sorted_byrows[2]:
                #         if pt[0] >= sorted_bycolumns[8]:
                #             right_knee_match = [pt[1], pt[0]]
                #         else:
                #             right_hip_match = [pt[1]+30, pt[0]]
                #             # print('Found left hip: ', left_hip_match)
                rt = template_img.shape[0]
                ct = template_img.shape[1]
                right_hip_match = [rt/2+height/4,ct/2-width/4]
                right_knee_match = [rt/2+height/4,ct/2+width/4]
                right_ankle_match = [rt,ct/2+width/4]
                try:
                    matched_points = np.array([right_hip_match,right_knee_match,right_ankle_match]).astype(np.float32)
                    key_bodypts = np.array([[right_hip[1],right_hip[0]],[right_knee[1],right_knee[0]],[right_ankle[1],right_ankle[0]]]).astype(np.float32)
                except:
                    print('Landmarks Not Found')
                    cv2.imshow("Image", output)
                    cv2.waitKey(1)
                    continue

            # for a in pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10:
            #     #print(a)
            #     x = int(a[0])
            #     y = int(a[1])
            #     #print(x, y)
            #     cv2.circle(chair_edges, (x,y), 5, (255,0,0), cv2.FILLED)
            #     cv2.imshow('Template',chair_edges)
            #     cv2.waitKey(1)

            # for y, x in matched_points:
            #     x = int(x)
            #     y = int(y)
            #     cv2.circle(template_img, (x,y), 5, (0,255,0), cv2.FILLED)
            #     cv2.circle(img, (x,y), 5, (0,255,0), cv2.FILLED)

            # for y,x in key_bodypts:
            #     x = int(x)
            #     y = int(y)
            #     cv2.circle(template_img, (x,y), 5, (0,0,255), cv2.FILLED)
            #     cv2.circle(img, (x,y), 5, (0,0,255), cv2.FILLED)

            # cv2.imshow('Superimposed',img)
            # cv2.imshow('Dots',template_img)
            # cv2.waitKey(1)

            # print(key_bodypts)
            # print(matched_points)

            warp = cv2.getAffineTransform(key_bodypts,matched_points)
            warped_template = cv2.warpAffine(template_img,warp,[img.shape[1],img.shape[0]])

            for y,x in key_bodypts:
                x = int(x)
                y = int(y)
                cv2.circle(warped_template, (x,y), 5, (0,0,255), cv2.FILLED)

            output = img.copy()
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if not (warped_template[i][j][0] == warped_template[i][j][1] == warped_template[i][j][2]):
                        output[i][j] = img[i][j] * .5 + warped_template[i][j] * .5 
        
        output = cv2.resize(output,[original_c,original_r])

        cv2.imshow("Image", output)
        cv2.waitKey(1)