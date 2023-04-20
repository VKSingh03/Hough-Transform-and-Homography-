import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import sympy as sym
from scipy.spatial.transform import Rotation
from itertools import combinations

eul_ang = []
transl = []

def filter(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)#converting image to HSV color scale
    hsv = cv.GaussianBlur(hsv,(5,5),0)
    lower_white = np.array([50,5,200],np.uint8)
    upper_white = np.array([110,110,255],np.uint8)
    mask = cv.inRange(hsv,lower_white,upper_white) #creating white mask
    mask = cv.erode(mask,None,iterations=5) #Eroding for removing noise
    mask = cv.dilate(mask, None, iterations=1) #dilating mask for better result
    ret, thresh = cv.threshold(mask,100,255, cv.THRESH_BINARY) #converting image to binary
    canny = cv.Canny(thresh, 200, 240)
    return(canny)

def hough_transform(img):
    H=dict()
    coordinates=np.where(img>0)
    coordinates=np.transpose(coordinates)
    for point in coordinates:
        for t in range(180):
            d=point[0]*np.sin(np.deg2rad(t))+point[1]*np.cos(np.deg2rad(t))
            d=int(d)
            if d<int(np.ceil(np.sqrt(np.square(img.shape[0])+np.square(img.shape[1])))):
                if (d,t) in H:
                    H[(d,t)] += 1
                else:
                    H[(d,t)] = 1
    return H
    
def computeSVD(mat):   
    m, n = mat.shape
    T1 = np.dot(mat, mat.transpose())
    T2 = np.dot(mat.transpose(), mat)
    ev1, U = np.linalg.eig(T1)
    ev2, V = np.linalg.eig(T2)
    # sort the eigen values and vectors
    # U matrix
    idx1 = np.flip(np.argsort(ev1))
    ev1 = ev1[idx1]
    U = U[:, idx1]
    # V matrix
    idx2 = np.flip(np.argsort(ev2))
    ev2 = ev2[idx2]
    V = V[:, idx2]
    # E matrix
    E = np.zeros([m, n])
    var = np.minimum(m, n)
    for j in range(var):
        E[j,j] = np.abs(np.sqrt(ev1[j]))  
    return U, E, V
    
def computeHomography(set1, set2):
    x = set1[:, 0]
    y = set1[:, 1]
    xp = set2[:, 0]
    yp = set2[:,1]
    A = []
    for i in range(int(4)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)
    A = np.array(A)
    U, E, V = computeSVD(A)
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    # print("the Homography matrix is")
    # print(H)
    # print(" The homography from cv function is:", cv.findHomography(set1, set2))
    return H

def trans_rot(H, K):
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    r1 = np.matmul(np.linalg.inv(K), h1) 
    r2 = np.matmul(np.linalg.inv(K), h2) 
    r3 = np.cross(r1, r2) 

    t = np.matmul(np.linalg.inv(K), h3) / np.linalg.norm(r1) 
    R = np.column_stack((r1, r2, r3)) 
    # print("Translation matrix: \n", t) 
    # print("Rotation matrix: \n", R) 
    r = Rotation.from_matrix(R) 
    euler_angles = r.as_euler('zyx', degrees=True) 
    # print("Euler angles", euler_angles) 
    # print("Translation matrix", t)
    eul_ang.append(euler_angles.tolist())
    transl.append(t.tolist())

def corners_select(sorted_hough_space_unfiltered):
    unique=dict()
    d, theta =[], []
    val, p_d, p_th =0,0,0
    t_th=10
    t_d=50
    for key, value in sorted_hough_space_unfiltered.items():
        if val==0:
            p_d=key[0]
            p_th=key[1]
            d.append(key[0])
            theta.append(key[1])
            val=value
        else:
            if p_d-t_d <= key[0] <= p_d+t_d and p_th-t_th <= key[1] <= p_th+t_th:
                val+=value
                p_d=key[0]
                p_th=key[1]
                d.append(key[0])
                theta.append(key[1])
            else:
                unique[(int(np.mean(d)),int(np.mean(theta)))]=val
                d=[]
                theta=[]
                val=0
    if d and val:
        unique[(int(np.mean(d)),int(np.mean(theta)))]=val
    return unique 

def corners(d1, theta1, d2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([d1, d2])
    x, y = np.linalg.solve(A, b)
    return [int(x), int(y)]

def extract_image(video, points_world, K):
    while True:
        isTrue, frame = video.read()
        if isTrue == True:
            frame1 = filter(frame)
            accumulator = dict(sorted(hough_transform(frame1).items(), key=lambda item: item[1],reverse=True)[:20])
            accumulator = dict(sorted(accumulator.items()))
            points=corners_select(accumulator)
            points=dict(sorted(points.items(), key=lambda item: item[1],reverse=True)[:4])
            comb = list(combinations(points.items(), 2))
            solution=[]
            filter_int=[]
            for (key1, value1), (key2, value2) in comb:
                try:
                    solution.append(corners(key1[0],np.deg2rad(key1[1]), key2[0],np.deg2rad(key2[1]))) 
                except:
                    pass
            for x,y in solution:
                if x>0 and y>0:
                    filter_int.append([x,y])
            points_frame = np.array(filter_int, dtype=np.float32)
            points_frame = points_frame[points_frame[:,0].argsort()]
            points_world = np.array([[0, 279], [216, 279], [0, 0], [216, 0]], dtype=np.float32)
            H=computeHomography(points_frame,points_world)
            trans_rot(H, K)    
            cv.imshow("Corners",frame1)
        else: 
            break 
        if cv.waitKey(20) & 0xFF==ord('c'): 
            break

def main(): 
    # Reading the video 
    video = cv.VideoCapture('project2.avi') 

    # The camera intrinsic matrix 
    K = np.array([[1382.58398, 0, 945.743164], [0, 1383.57251, 527.04834], [0, 0, 1]], dtype=np.float32) 

    # Actual size of paper in world frame converted to mm 
    Points_world = np.array([[0,0], [0,216], [279,216], [279,0]]) 

    # Extraccting frame and computing Homography
    extract_image(video, Points_world, K) 

    # data = eul_ang
    # 
    # eul_ang = np.array(eul_ang)
    # transl = np.array(transl)
    # x = range(len(eul_ang))

    # # Plot the data
    # plt.legend()
    # plt.title('Rotation plot')
    # plt.plot(range(len(eul_ang)),eul_ang)
    # plt.show()
    # plt.title('Translation plot')
    # plt.plot(range(len(transl)),transl)
    # plt.show()

    euler_angles=np.array(eul_ang)

    # Create a new figure
    fig, ax = plt.subplots()
    names=['Z','Y','X']
    # Plot each column as a line
    for i in range(euler_angles.shape[1]):
        ax.plot(euler_angles[:,i], label=f"{names[i]}")

    # Add legend and axis labels
    plt.legend(loc='upper left')
    ax.set_xlabel("No of frames")
    ax.set_ylabel("angles")
    ax.set_title("Euler Angles")

    # Show the plot
    plt.show()

    translation=np.array(transl)

    # Create a new figure
    fig, ax = plt.subplots()
    t_names=['X','Y','Z']
    # Plot each column as a line
    for i in range(translation.shape[1]):
        ax.plot(translation[:,i], label=f"{names[i]}")

    # Add legend and axis labels
    plt.legend(loc='upper left')
    ax.set_xlabel("No of frames")
    ax.set_ylabel("distance")
    ax.set_title("Translation Matrix")

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    main()