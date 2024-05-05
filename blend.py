import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# constants for thresholds
match_quality_threshold = 0.02
inlier_threshold = 0.5
homography_inlier_threshold = 0.55
ratio = 0.7
homography_error = 5.0

# use SIFT to get kp and desc
def find_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# use FLANN to match keypoints
def match_keypoints(desc1, desc2):
    # use FLANN params for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # do the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append(m)
    return good_matches

# use RANSAC to get the fundamental matrix
def find_fundamental_matrix(matches, keypoints1, keypoints2):
    if len(matches) < 8:
        return None, None
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    return F, mask

# create a mosaic based on homography
def create_mosaic(img1, img2, H):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # get corners, then transform
    corners_img1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners_img2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    corners_img2_transformed = cv2.perspectiveTransform(corners_img2, H)
    
    # combine to get mosaic size
    all_corners = np.concatenate((corners_img1, corners_img2_transformed), axis=0)
    
    [x_mi, y_mi] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_ma, y_ma] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # change H translation so it shifts to positive coords
    translation_dist = [-x_mi, -y_mi]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # warp images
    output_size = (x_ma - x_mi, y_ma - y_mi)
    img2_warped = cv2.warpPerspective(img2, H_translation.dot(H), output_size)
    
    # creat mosaic
    mosaic = np.zeros((y_ma - y_mi, x_ma - x_mi, 3), dtype=np.uint8)
    mosaic[translation_dist[1]:height1+translation_dist[1], translation_dist[0]:width1+translation_dist[0]] = img1
    
    # mask for overlapping regions
    overlap = np.where((mosaic != 0) & (img2_warped != 0))

    # average the overlaps for a better blend
    mosaic[overlap] = np.average([mosaic[overlap], img2_warped[overlap]], axis=0)

    # add img2 to the black (unfilled) pixels
    black = np.all(mosaic == 0, axis=-1)
    mosaic[black] = img2_warped[black]

    return mosaic
    
# draw epipolar lines
def epipolar_lines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    # put images in color
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # get start and end each line
    x0 = np.zeros(lines.shape[0], dtype=int)
    y0 = np.clip((-lines[:, 2] / lines[:, 1]).astype(int), 0, r - 1)
    x1 = np.full(lines.shape[0], c, dtype=int)
    y1 = np.clip((-(lines[:, 2] + lines[:, 0] * c) / lines[:, 1]).astype(int), 0, r - 1)

    # gen random line colors
    colors = np.random.randint(0, 255, (lines.shape[0], 3)).astype(int)

    # draw the lines and points on the images
    for (x0_, y0_, x1_, y1_, color, pt1, pt2) in zip(x0, y0, x1, y1, colors, pts1, pts2):
        color_tuple = tuple(color.tolist()) # Ensure color is a tuple of integers
        img1 = cv2.line(img1, (x0_, y0_), (x1_, y1_), color_tuple, 1)
        img1 = cv2.circle(img1, tuple(np.round(pt1).astype(int)), 5, color_tuple, -1)
        img2 = cv2.circle(img2, tuple(np.round(pt2).astype(int)), 5, color_tuple, -1)

    return img1

# driver func to process a pair of images
def process_image_pair(im_path1, im_path2, out_dir='./output'):
    # load the two images
    img1 = cv2.cvtColor(cv2.imread(im_path1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(im_path2), cv2.COLOR_BGR2GRAY)
    
    # extract the keypoints and descriptors of each img
    kp1, des1 = find_keypoints_and_descriptors(img1)
    kp2, des2 = find_keypoints_and_descriptors(img2)
    print(f"\tImage 1: {len(kp1)} keypoints, Image 2: {len(kp2)} keypoints")
    
    # match the keypoints
    matches = match_keypoints(des1, des2)
    match_percentage = len(matches)/min(len(kp1), len(kp2))
    print(f"\tMatches: {len(matches)}, Match percentage: {match_percentage:.2f}")
    
    # visualize with cv2.drawMatches using different colors
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Matches")
    plt.show()

    # check match percentage
    if match_percentage < match_quality_threshold:
        print("\tLikely different scenes. Terminating...")
        return
    else:
        print("\tLikely same scene. Proceeding...")
    
    # estimate fundamental matrix
    F, mask = find_fundamental_matrix(matches, kp1, kp2)
    if F is None:
        print("\tCould not compute fundamental matrix.")
        return
    
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i,0]]
    print(f"\tInliers: {len(inlier_matches)}, Inlier percentage: {len(inlier_matches)/len(matches):.2f}")
    
    # display with lines drawn between keypoints that form inlier matches (cv2.drawMatches)
    inlier_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(inlier_img, cv2.COLOR_BGR2RGB)), plt.title("Inlier Matches"), plt.show()

    # show epipolar lines for inlier matches drawn on img2
    lines = cv2.computeCorrespondEpilines(np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1,1,2), 1, F)
    lines = lines.reshape(-1, 3)
    epi_img = epipolar_lines(img2.copy(), img1.copy(), lines, np.float32([kp1[m.queryIdx].pt for m in inlier_matches]), np.float32([kp2[m.trainIdx].pt for m in inlier_matches]))
    plt.imshow(cv2.cvtColor(epi_img, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines")
    plt.show()


    # check inlier percentage
    if len(inlier_matches) < 8 or len(inlier_matches)/len(matches) < inlier_threshold:
        print("\tToo few inlier matches. Terminating...")
        return
    else:
        print("\tGood number of inlier matches. Proceeding...")

    # estimate homography matrix
    H, status = cv2.findHomography(np.float32([kp2[m.trainIdx].pt for m in inlier_matches]), 
                           np.float32([kp1[m.queryIdx].pt for m in inlier_matches]), 
                           cv2.RANSAC, homography_error)

    homography_inliers = [m for m, s in zip(inlier_matches, status.ravel()) if s == 1]
    homography_percentage = len(homography_inliers)/len(inlier_matches)
    print(f"\tHomography inliers: {len(homography_inliers)}, Homography percentage: {homography_percentage:.2f}")

    # display homography image
    homog_img = cv2.drawMatches(img1, kp1, img2, kp2, homography_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(homog_img, cv2.COLOR_BGR2RGB))
    plt.title("Homography Inlier Matches")
    plt.show()
    
    # check homography percentage
    if homography_percentage < homography_inlier_threshold:
        print("\tToo small of a homography percentage. Terminating...")
        return
    else:
        print("\tAcceptable homography percentage. Proceeding...")

    
    # create and save the mosaic
    mosaic = create_mosaic(cv2.imread(im_path1), cv2.imread(im_path2), H)

    # name mosaic img1_img2 with extension of img1
    im_name1 = im_path1.split('/')[-1]
    im_name2 = im_path2.split('/')[-1]
    mosaic_name = im_name1[:-4] + '_' + im_name2[:-4] + im_name1[-4:]

    print(f"\tSaving {mosaic_name} to {out_dir}")

    # write the mosaic to out_dir
    cv2.imwrite(os.path.join(out_dir, mosaic_name), mosaic)

# MAIN
def main(in_dir, out_dir):
    # check if in_dir exists
    if in_dir == 'err' or not os.path.exists(in_dir):
        print(f"Directory {in_dir} does not exist. Terminating...")
        return
    
    # read in image paths
    images = [f for f in os.listdir(in_dir) if f.endswith('.JPG') or f.endswith('.jpg')]
    n = len(images)
    print(f"Found {n} images in {in_dir}...")

    # create out_dir if necessary
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # go through each image pair
    for i in range(n):
        for j in range(i + 1, n):
            image_path1 = os.path.join(in_dir, images[i])
            image_path2 = os.path.join(in_dir, images[j])
            print(f"Processing {images[i]} and {images[j]}")
            process_image_pair(image_path1, image_path2, out_dir)


# read arguments
if __name__ == "__main__":
    import sys
    in_dir = sys.argv[1] if len(sys.argv) > 1 else 'err'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    main(in_dir, out_dir)