import cv2
import numpy as np
import pickle
import cv2.aruco

# ----------------------------
# Helper functions
# ----------------------------
def resize_to_screen(img, screen_width=1366, screen_height=768):
    max_width = int(screen_width * 0.75)
    max_height = int(screen_height * 0.75)
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

# ----------------------------
# Trackbar callback
# ----------------------------
def nothing(x):
    pass

# ----------------------------
# Safe destroy for windows
# ----------------------------
def safe_destroy(window_name):
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass



# ----------------------------
# Transformations (Translation, Rotation, Scaling)
# ----------------------------

# ----------------------------
# Transformations (All-in-one: Translation, Rotation, Scaling)
# ----------------------------
def create_transform_trackbars():
    cv2.namedWindow("Webcam")
    cv2.createTrackbar("Tx", "Webcam", 200, 400, nothing)   # centered at 0
    cv2.createTrackbar("Ty", "Webcam", 200, 400, nothing)   # centered at 0
    cv2.createTrackbar("Angle", "Webcam", 180, 360, nothing) # centered at 0
    cv2.createTrackbar("Scale", "Webcam", 100, 300, nothing) # 100 = 1.0x

def apply_all_transform(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Read trackbars
    tx = cv2.getTrackbarPos("Tx", "Webcam") - 200
    ty = cv2.getTrackbarPos("Ty", "Webcam") - 200
    angle = cv2.getTrackbarPos("Angle", "Webcam") - 180
    scale_val = cv2.getTrackbarPos("Scale", "Webcam") / 100.0

    # Rotation + Scale matrix
    M = cv2.getRotationMatrix2D(center, angle, scale_val)
    # Add translation
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(47, 53, 66))


camera_matrix = None
dist_coeffs = None
calibrated = False

# ----------------------------
# Calibrate Camera
# ----------------------------


def calibrate_camera_from_webcam(cap, chessboard_size=(9,6), num_frames=20):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    collected = 0
    gray = None

    while collected < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
            collected += 1
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)
            cv2.putText(frame, f"Captured {collected}/{num_frames}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Show chessboard...", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    safe_destroy("Calibration")   # <-- safer than destroyWindow

    if len(objpoints) > 0:
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    else:
        return None, None

    


# ----------------------------
# Side-by-side distortion demo
# ----------------------------


def show_distortion_demo(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

    frame_resized = cv2.resize(frame, (w//2, h//2))
    undist_resized = cv2.resize(undistorted, (w//2, h//2))

    comparison = np.hstack((frame_resized, undist_resized))

    cv2.putText(comparison, "Distorted", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(comparison, "Undistorted", (w//2+30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return comparison




# ----------------------------
# Panorama functions
# ----------------------------
def capture_frames(num_frames=5):
    cap = cv2.VideoCapture(0)
    frames = []
    count = 0
    print("Press 'q' to capture a frame")
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            frames.append(frame)
            count += 1
            print(f"Captured frame {count}")
    cap.release()
    cv2.destroyAllWindows()
    return frames

def stitch_pair(img1, img2):
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        print("Not enough matches to stitch")
        return img1

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    H_translate = np.array([[1,0,translation[0]], [0,1,translation[1]], [0,0,1]])

    panorama = cv2.warpPerspective(img1, H_translate @ H, (xmax-xmin, ymax-ymin))
    panorama[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = img2
    return panorama

def stitch_frames(frames):
    panorama = frames[0]
    for i in range(1, len(frames)):
        print(f"Stitching frame {i+1}/{len(frames)} ...")
        panorama = stitch_pair(panorama, frames[i])
    return panorama

def show_resized(window_name, img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    cv2.imshow(window_name, resized)


# ----------------------------
# Save/Load calibration
# ----------------------------

def save_calibration(filename, camera_matrix, dist_coeffs):
    with open(filename, 'wb') as f:
        pickle.dump({'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}, f)

def load_calibration(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data['camera_matrix'], data['dist_coeffs']
    except Exception:
        return None, None


# ----------------------------
# ARUCO and 3D Model functions  
# ----------------------------

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), faces

def project_obj(img, vertices, faces, rvec, tvec, camera_matrix, dist_coeffs):
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.squeeze().astype(int)
    for face in faces:
        pts = imgpts[face]
        cv2.polylines(img, [pts], True, (0,255,0), 1)
    return img

# Load OBJ once
trex_vertices, trex_faces = load_obj("trex_model.obj")  # Change filename as needed
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
aruco_params = cv2.aruco.DetectorParameters()
marker_length = 0.14  # meters

# In main loop:


# ----------------------------
# Initialize webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

mode = "normal"
color_mode = "rgb"
hsv_channel = "all"
filter_mode = None
edge_mode = None
trackbars_created = False
transform_mode = None
trackbars_created_contrast = False

cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 1024, 768)

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    display = frame.copy()

    # ----------------------------
    # EDGE MODE
    # ----------------------------
    if mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if edge_mode == "canny":
            edges = cv2.Canny(gray, 50, 150, L2gradient=True)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif edge_mode == "line":
            edges = cv2.Canny(gray, 50, 150, L2gradient=True)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
            display = frame.copy()
            if lines is not None:
                for rho, theta in lines[:,0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(display, (x1,y1), (x2,y2), (0,0,255), 2)

        # Resize for display
        display_resized = resize_to_screen(display)
        h, w = display_resized.shape[:2]
        canvas = np.zeros((h+25, w,3), dtype=np.uint8)
        canvas[:h,:,:] = display_resized

        # Instructions inside edge mode
        if edge_mode is None:
            instructions = "[c] Canny Edge  [l] Hough Lines  [ESC] home"
        elif edge_mode == "canny":
            instructions = "Canny Edge  [l] switch to Lines  [ESC] home"
        elif edge_mode == "line":
            instructions = "Hough Lines  [c] switch to Canny  [ESC] home"

        cv2.putText(canvas, instructions, (5, h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA)
        cv2.imshow("Webcam", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            mode = "normal"
            edge_mode = None
        elif key == ord('c'):
            edge_mode = "canny"
        elif key == ord('l'):
            edge_mode = "line"
        continue

    

    # ----------------------------
    # Display home / main screen
    # ----------------------------
    display_resized = resize_to_screen(display)
    h, w = display_resized.shape[:2]
    canvas = np.zeros((h+50, w,3), dtype=np.uint8)
    canvas[:h,:,:] = display_resized



        # ----------------------------
    # FILTER MODE
    # ----------------------------
    if mode == "filter":
        if filter_mode is None:
            menu = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(menu, "Filter Mode", (80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(menu, "[g] Gaussian Filter", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(menu, "[b] Bilateral Filter", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(menu, "[ESC] Back Home", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Webcam", menu)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                filter_mode = "gaussian"
                trackbars_created = False
            elif key == ord('b'):
                filter_mode = "bilateral"
                trackbars_created = False
            elif key == 27:
                mode = "normal"
                filter_mode = None
                trackbars_created = False
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if filter_mode == "gaussian":
            if not trackbars_created:
                cv2.namedWindow("Webcam")
                cv2.createTrackbar("Kernel Size", "Webcam", 5, 21, nothing)
                cv2.createTrackbar("Sigma", "Webcam", 10, 100, nothing)
                trackbars_created = True

            k = cv2.getTrackbarPos("Kernel Size", "Webcam")
            sigma = cv2.getTrackbarPos("Sigma", "Webcam") / 10.0
            if k % 2 == 0: k += 1
            if k < 1: k = 1
            gaussian = cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma)
            g_disp = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR)
            display_resized = resize_to_screen(g_disp)
            h, w = display_resized.shape[:2]
            canvas = np.zeros((h+50, w,3), dtype=np.uint8)
            canvas[:h,:,:] = display_resized
            cv2.putText(canvas, f"Gaussian Filter: k={k}, sigma={sigma:.1f}  [f] filter menu  [ESC] home", (10, h+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Webcam", canvas)

        elif filter_mode == "bilateral":
            if not trackbars_created:
                cv2.namedWindow("Webcam")
                cv2.createTrackbar("d (Diameter)", "Webcam", 9, 20, nothing)
                cv2.createTrackbar("sigmaColor", "Webcam", 75, 150, nothing)
                cv2.createTrackbar("sigmaSpace", "Webcam", 75, 150, nothing)
                trackbars_created = True

            d = cv2.getTrackbarPos("d (Diameter)", "Webcam")
            sigmaColor = cv2.getTrackbarPos("sigmaColor", "Webcam")
            sigmaSpace = cv2.getTrackbarPos("sigmaSpace", "Webcam")
            if d <= 0: d = 1
            bilateral = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)
            b_disp = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
            display_resized = resize_to_screen(b_disp)
            h, w = display_resized.shape[:2]
            canvas = np.zeros((h+50, w,3), dtype=np.uint8)
            canvas[:h,:,:] = display_resized
            cv2.putText(canvas, f"Bilateral: d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}", (10, h+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, "[f] filter menu  [ESC] home", (10, h+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Webcam", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            mode = "normal"
            filter_mode = None
            trackbars_created = False
            safe_destroy("Webcam")
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam", 1024, 768)
        elif key == ord('g'):
            filter_mode = "gaussian"
            trackbars_created = False
        elif key == ord('b'):
            filter_mode = "bilateral"
            trackbars_created = False
        elif key == ord('f'):
            filter_mode = None
            trackbars_created = False
            safe_destroy("Webcam")
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam", 1024, 768)
        continue

    
    # ----------------------------
    # COLOR MODE
    # ----------------------------
    if mode == "color":
        if color_mode == "rgb":
            display = frame.copy()
        elif color_mode == "gray":
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ----------------------------
    # HSV MODE (sub-mode of color)
    # ----------------------------
    if mode == "hsv":
        hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
        if hsv_channel == "hue":
            hue = hsv[..., 0]
            hue_vis = cv2.normalize(hue, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.applyColorMap(hue_vis, cv2.COLORMAP_HSV)
        elif hsv_channel == "saturation":
            sat = hsv[..., 1]
            sat_vis = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.applyColorMap(sat_vis, cv2.COLORMAP_JET)
        elif hsv_channel == "value":
            val = hsv[..., 2]
            val_vis = cv2.normalize(val, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.applyColorMap(val_vis, cv2.COLORMAP_JET)
        else:
            display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        

    

    if mode == "distortion_demo":
        if not calibrated:
            camera_matrix, dist_coeffs = calibrate_camera_from_webcam(cap)
            if camera_matrix is not None:
                calibrated = True
                save_calibration("calibration.pkl", camera_matrix, dist_coeffs)  # Save calibration
                print("Calibration successful!")
            else:
                print("Calibration failed.")
                mode = "normal"
        else:
            ret, frame = cap.read()
            if not ret: break
            comparison = show_distortion_demo(frame, camera_matrix, dist_coeffs)
            cv2.imshow("Webcam", comparison)
            instructions = "[ESC] back  [q] quit"



    # ----------------------------
    # Transform mode
    # ----------------------------
    if mode == "transform":

        if not trackbars_created:
            create_transform_trackbars()
            trackbars_created = True

        display = apply_all_transform(frame)

        display_resized = resize_to_screen(display)
        h, w = display_resized.shape[:2]
        canvas = np.zeros((h+25, w,3), dtype=np.uint8)
        canvas[:h,:,:] = display_resized

        instructions = "Transform Mode: Adjust sliders (Tx, Ty, Angle, Scale)  [r] reset  [ESC] home"
        cv2.putText(canvas, instructions, (5,h+18), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255),1,cv2.LINE_AA)
        cv2.imshow("Webcam", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC
            mode = "normal"
            safe_destroy("Webcam")  # Remove trackbars
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam", 1024, 768)
        elif key == ord('r'):
            # Reset trackbars to defaults
            cv2.setTrackbarPos("Tx", "WebCam", 200)
            cv2.setTrackbarPos("Ty", "WebCam", 200)
            cv2.setTrackbarPos("Angle", "WebCam", 180)
            cv2.setTrackbarPos("Scale", "WebCam", 100)
        #continue

    # CONTRAST MODE
    if mode == "contrast":
        if not trackbars_created_contrast:
            cv2.createTrackbar("Contrast", "Webcam", 50, 100, nothing)
            cv2.createTrackbar("Brightness", "Webcam", 100, 200, nothing)
            trackbars_created_contrast = True
        alpha = cv2.getTrackbarPos("Contrast", "Webcam") / 50.0
        beta = cv2.getTrackbarPos("Brightness", "Webcam") - 100
        display = cv2.convertScaleAbs(display, alpha=alpha, beta=beta)

        display_resized = resize_to_screen(display)
        h, w = display_resized.shape[:2]        
        canvas = np.zeros((h+25, w,3), dtype=np.uint8)
        canvas[:h,:,:] = display_resized
        instructions = "Adjust contrast & brightness  [ESC] home"
        cv2.putText(canvas, instructions, (5,h+18), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255),1,cv2.LINE_AA)
        cv2.imshow("Webcam", canvas)

    if mode == "ar":
        instructions = "Show A4_ArUco_Marker.png to camera  [ESC] home"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        display = frame.copy()
        if ids is not None and calibrated:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(display, camera_matrix, dist_coeffs, rvec, tvec, 0.07)
                display = project_obj(display, trex_vertices, trex_faces, rvec, tvec, camera_matrix, dist_coeffs)
        display_resized = resize_to_screen(display)
        h, w = display_resized.shape[:2]
        canvas = np.zeros((h+25, w,3), dtype=np.uint8)
        canvas[:h,:,:] = display_resized
        cv2.putText(canvas, instructions, (5,h+18), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255),1,cv2.LINE_AA)
        cv2.imshow("Webcam", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            mode = "normal"
        continue


    # ----------------------------
    # HISTOGRAM MODE
    # ----------------------------
    if mode == "histogram":
        #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        img_gray = display.copy()  # use current color mode display
        #img_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    # ... continue histogram drawing as before


        hist_h, hist_w = 256, 256
        hist_canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            hist = hist.flatten()
            for x in range(256):
                y = int(hist[x])
                cv2.line(hist_canvas, (x, hist_h), (x, hist_h - y),
                         (255 if col == 'r' else 0,
                          255 if col == 'g' else 0,
                          255 if col == 'b' else 0))

        display_resized = resize_to_screen(frame)
        h, w = display_resized.shape[:2]
        canvas = np.zeros((h + hist_h + 50, max(w, hist_w), 3), dtype=np.uint8)
        canvas[:h, :w, :] = display_resized
        canvas[h:h + hist_h, :hist_w, :] = hist_canvas

        cv2.putText(canvas, "Press ESC to go home", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Webcam", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            mode = "normal"
            color_mode = "rgb"
            hsv_channel = "all"
        continue

    # ----------------------------
    # Resize display to screen
    # ----------------------------
    display_resized = resize_to_screen(display)

    # ----------------------------
    # Add instruction bars
    # ----------------------------
    h, w = display_resized.shape[:2]
    top_bar_height = 25
    bottom_bar_height = 25
    canvas = np.zeros((h + top_bar_height + bottom_bar_height, w, 3), dtype=np.uint8)
    canvas[top_bar_height:h + top_bar_height, :, :] = display_resized

    

    # Two-line instructions
    if mode == "normal":
        instructions1 = "[i] image color  [c] contrast  [h] histogram  [f] filter"
        instructions2 = "[e] edge  [p] panorama  [m] transform [d] calibrate [a] AR [q] quit"
    elif mode == "filter":
        instructions = "[g] Gaussian  [b] Bilateral  [ESC] home"
    elif mode == "contrast":
        instructions1 = "Adjust contrast & brightness  [ESC] home"
        instructions2 = ""
    elif mode == "color":
        instructions1 = "[g] Gray  [h] HSV  [r] RGB  [ESC] home"
        instructions2 = ""
    elif mode == "hsv":
        instructions1 = "[0] Hue  [1] Saturation  [2] Value  [ESC] home"
        istructions2 = ""
    else:
        instructions1 = "Press ESC to go home"
        instructions2 = ""

    cv2.putText(canvas, instructions1, (5,h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA)
    cv2.putText(canvas, instructions2, (5,h+38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA)
    #cv2.putText(canvas, instructions, (5, h + top_bar_height + 18),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Webcam", canvas)

    # ----------------------------
    # Home key controls
    # ----------------------------

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 27:
        if mode == "hsv":
            mode = "color"       # back to parent color mode
            hsv_channel = "all"

        elif mode == "contrast":
            mode = "normal"
            trackbars_created_contrast = False
            safe_destroy("Webcam")  # This removes the trackbars
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam", 1024, 768)
        elif mode == "filter":
            mode = "normal"
            filter_mode = None
            trackbars_created = False
            safe_destroy("Webcam")  # Remove filter trackbars
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam", 1024, 768)
        else:
            mode = "normal"
            color_mode = "rgb"
            hsv_channel = "all"
            filter_mode = None
            edge_mode = None
            trackbars_created = False
            safe_destroy("Gaussian Filter")
            safe_destroy("Bilateral Filter")
            safe_destroy("Filter Selection")
    elif key == ord('i'):
        mode = "color"
    elif key == ord('e'):
        mode = "edge"
        edge_mode = None
    elif key == ord('p'):
        cap.release()
        cv2.destroyAllWindows()
        print("Entering Panorama Mode...")
        frames = capture_frames(5)
        if len(frames) >= 2:
            pano = stitch_frames(frames)
            cv2.imwrite("panorama.jpg", pano)
            print("Panorama saved as 'panorama.jpg'")
            show_resized("Panorama", pano)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap = cv2.VideoCapture(0)
        mode = "normal"
    elif key == ord('c') and mode == "normal":
        mode = "contrast"
        #trackbars_created = False
    elif key == ord('h') and mode == "normal":
        mode = "histogram"
    elif key == ord('m'):
        mode = "transform"
        safe_destroy("Transform Controls")   # ensure clean start
        trackbars_created = False
        #create_transform_trackbars()
    elif key == ord('f') and mode == "normal":
        mode = "filter"
        filter_mode = None
        trackbars_created = False
    elif mode == "color":
    # Color mode selections: do NOT leave color mode
        if key == ord('r'):
            color_mode = "rgb"
        elif key == ord('g'):
            color_mode = "gray"
        elif key == ord('h'):
            color_mode = "hsv"
            hsv_channel = "all"
            mode = "hsv"
    elif mode == "hsv":
    # HSV subchannels
        if key == ord('0'):
            hsv_channel = "hue"
        elif key == ord('1'):
            hsv_channel = "saturation"
        elif key == ord('2'):
            hsv_channel = "value"
    elif key == ord('d') and mode == "normal":
        mode = "distortion_demo"
    elif key == ord('a'):
        mode = "ar"

      


cap.release()
cv2.destroyAllWindows()
