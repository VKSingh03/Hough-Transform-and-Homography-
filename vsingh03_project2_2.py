import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

image_paths=['image_1.jpg','image_2.jpg','image_3.jpg','image_4.jpg']
# initialized a list of images
imgs = []

def read_image(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray, img_rgb

def SIFT(img):
    sift= cv2.SIFT_create()
    print("Inside sift")
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    print("Inside plotsift")
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def matcher(kp1, des1, kp2, des2, thresh):
    # BFMatcher function
    bf = cv2.BFMatcher()
    print("Matching features")
    matches = bf.knnMatch(des1,des2,k=2)
    # print("Inside matcher")
    # Checking if within threshold
    selected = []
    for m,n in matches:
        if m.distance < thresh*n.distance:
            selected.append([m])

    matches = []
    for select in selected:
        matches.append(list(kp1[select[0].queryIdx].pt + kp2[select[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    print("Inside plot matches")
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    # plt.show()

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=8):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

def main():
    
    first_gray, first_rgb = read_image(image_paths[1])
    second_gray, second_rgb = read_image(image_paths[0])
    third_gray, third_rgb = read_image(image_paths[3])
    fourth_gray, fourth_rgb = read_image(image_paths[2])

    kp_first, des_first = SIFT(first_gray)
    kp_second, des_second = SIFT(second_gray)
    kp_third, des_third = SIFT(third_gray)
    kp_fourth, des_fourth = SIFT(fourth_gray)

    # kp_left_img1 = plot_sift(gray=first_gray,rgb=first_rgb,kp=kp_first)
    # kp_right_img2 = plot_sift(gray=second_gray,rgb=second_rgb,kp=kp_second)
    # kp_left_img3 = plot_sift(gray=third_gray,rgb=third_rgb,kp=kp_third)
    # kp_right_img4 = plot_sift(gray=fourth_gray,rgb=fourth_rgb,kp=kp_first)

    # total_kp1 = np.concatenate((kp_left_img1, kp_right_img2), axis=1)
    # total_kp2 = np.concatenate((kp_left_img3, kp_right_img4), axis=1)
    # plt.imshow(total_kp1)
    # plt.imshow(total_kp2)

    matches12 = matcher(kp_first, des_first, kp_second, des_second, 0.5)
    # matches34 = matcher(kp_third, des_third, kp_fourth, des_fourth, 0.5)

    inliers12, H12 = ransac(matches12, 0.8, 2000)
    # inliers34, H34 = ransac(matches34, 0.8, 2000)

    plt.imshow(stitch_img(first_rgb, second_rgb, H12))
    # plt.imshow(stitch_img(third_rgb, fourth_rgb, H34))
    
    plt.show()

if __name__ == "__main__":
    main()