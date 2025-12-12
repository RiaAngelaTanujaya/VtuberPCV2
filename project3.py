import cv2
import mediapipe as mp
import numpy as np
import math

# --- SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands 

# --- PALET WARNA ---
BG_NORMAL = (40, 44, 52)
RABBIT_FUR = (252, 252, 252)
RABBIT_SHADOW = (200, 200, 205) # Sedikit lebih gelap lagi untuk kontras body
BELLY_COLOR = (245, 235, 255)
INNER_EAR = (255, 185, 210)
NOSE_COLOR = (255, 150, 150)
MOUTH_COLOR = (80, 40, 40)
EYE_COLOR = (40, 40, 40)
CHEEK_COLOR = (255, 210, 220)

VIS_THRESH = 0.5 

def calc_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def rotate_point(point, center, angle_deg):
    angle_rad = math.radians(angle_deg)
    ox, oy = center
    px, py = point
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return (int(qx), int(qy))

# --- ENGINE GAMBAR RABBIT V10 (FINAL) ---
def draw_cute_rabbit_v10(canvas, results, shape):
    h, w, _ = shape
    pose = results.pose_landmarks
    face = results.face_landmarks

    # 1. KOORDINAT & ZOOM
    roi_lms = [lm for lm in pose.landmark if lm.visibility > VIS_THRESH]
    if not roi_lms and face: roi_lms = face.landmark
    if not roi_lms: return canvas 

    xs = [lm.x for lm in roi_lms]
    ys = [lm.y for lm in roi_lms]
    
    pad_x = (max(xs) - min(xs)) * 0.5
    pad_y_top = (max(ys) - min(ys)) * 1.5
    pad_y_bot = (max(ys) - min(ys)) * 0.8
    c_min_x, c_max_x = max(0, min(xs) - pad_x), min(1, max(xs) + pad_x)
    c_min_y, c_max_y = max(0, min(ys) - pad_y_top), min(1, max(ys) + pad_y_bot)

    scale = min(w / (c_max_x - c_min_x), h / (c_max_y - c_min_y)) * 0.75
    center_x_src, center_y_src = (c_min_x + c_max_x) / 2, (c_min_y + c_max_y) / 2

    def to_px(lm):
        sx, sy = (lm.x - center_x_src) * scale, (lm.y - center_y_src) * scale
        return (int(sx + w/2), int(sy + h/2))

    # 2. DATA UTAMA
    eye_l_lm, eye_r_lm = face.landmark[33], face.landmark[263]
    angle_tilt = -math.degrees(math.atan2(eye_r_lm.y - eye_l_lm.y, eye_r_lm.x - eye_l_lm.x))
    
    head_center = to_px(face.landmark[4]) 
    head_radius = int(calc_dist(face.landmark[152], face.landmark[10]) * scale * 0.6)

    # 3. TELINGA (Tetap butuh rotasi karena sifatnya 'nempel' dari atas kepala)
    ear_w, ear_h = int(head_radius * 0.35), int(head_radius * 1.6)
    def draw_ears(mult):
        base = (head_center[0] + int(head_radius*0.45)*mult, head_center[1] - int(head_radius*0.85))
        rot_c = rotate_point(base, head_center, angle_tilt)
        cv2.ellipse(canvas, rot_c, (ear_w+5, ear_h+5), angle_tilt, 0, 360, RABBIT_SHADOW, -1)
        cv2.ellipse(canvas, rot_c, (ear_w, ear_h), angle_tilt, 0, 360, RABBIT_FUR, -1)
        inner_c = rotate_point((base[0], base[1] + int(ear_h*0.2)), head_center, angle_tilt)
        cv2.ellipse(canvas, inner_c, (int(ear_w*0.6), int(ear_h*0.65)), angle_tilt, 0, 360, INNER_EAR, -1)
    draw_ears(-1); draw_ears(1)

    # 4. BADAN & LEHER
    lm_shldr_l, lm_shldr_r = pose.landmark[11], pose.landmark[12]
    lm_hip_l, lm_hip_r = pose.landmark[23], pose.landmark[24] # Landmark Hip

    pt_shldr_l = to_px(lm_shldr_l) if lm_shldr_l.visibility > VIS_THRESH else None
    pt_shldr_r = to_px(lm_shldr_r) if lm_shldr_r.visibility > VIS_THRESH else None
    pt_hip_l = to_px(lm_hip_l) if lm_hip_l.visibility > VIS_THRESH else None
    pt_hip_r = to_px(lm_hip_r) if lm_hip_r.visibility > VIS_THRESH else None

    if pt_shldr_l and pt_shldr_r:
        shoulder_dist = math.dist(pt_shldr_l, pt_shldr_r)
        shoulder_mid = ((pt_shldr_l[0]+pt_shldr_r[0])//2, (pt_shldr_l[1]+pt_shldr_r[1])//2)

        neck_base_w = int(shoulder_dist * 0.5)
        neck_pts = np.array([
            rotate_point((head_center[0]-neck_base_w//3, head_center[1]+int(head_radius*0.7)), head_center, angle_tilt),
            rotate_point((head_center[0]+neck_base_w//3, head_center[1]+int(head_radius*0.7)), head_center, angle_tilt),
            (shoulder_mid[0]+neck_base_w//2, shoulder_mid[1]),
            (shoulder_mid[0]-neck_base_w//2, shoulder_mid[1])
        ], np.int32)
        cv2.fillPoly(canvas, [neck_pts], RABBIT_FUR)

        if pt_hip_l and pt_hip_r:
            hip_mid = ((pt_hip_l[0]+pt_hip_r[0])//2, (pt_hip_l[1]+pt_hip_r[1])//2)
            body_center_y = (shoulder_mid[1] + hip_mid[1]) // 2 
            torso_len = math.dist(shoulder_mid, hip_mid) * 1.0 # Diperbesar dikit
        else: # Fallback jika hip tidak terdeteksi
            hip_mid = (shoulder_mid[0], shoulder_mid[1] + int(head_radius * 2.5))
            body_center_y = (shoulder_mid[1] + hip_mid[1]) // 2 
            torso_len = int(head_radius * 2.5)

        body_center = (shoulder_mid[0], body_center_y)
        body_w_radius = int(shoulder_dist * 0.9) 
        body_h_radius = int(torso_len * 0.65) # Body oval

        cv2.ellipse(canvas, body_center, (body_w_radius+5, body_h_radius+5), 0, 0, 360, RABBIT_SHADOW, -1)
        cv2.ellipse(canvas, body_center, (body_w_radius, body_h_radius), 0, 0, 360, RABBIT_FUR, -1)
        cv2.ellipse(canvas, (body_center[0], body_center[1] + int(body_h_radius*0.15)), 
                   (int(body_w_radius*0.6), int(body_h_radius*0.5)), 0, 0, 360, BELLY_COLOR, -1)

    # 5. KEPALA
    cv2.circle(canvas, head_center, head_radius+5, RABBIT_SHADOW, -1)
    cv2.circle(canvas, head_center, head_radius, RABBIT_FUR, -1)

    # 6. WAJAH (FIXED: HIDUNG, MULUT, PIPI TIDAK BEROTASI MANUAL)
    
    # Pipi (Cheek) - TIDAK DIPUTAR LAGI, hanya di-offset dari landmark
    cheek_offset_x = int(head_radius * 0.6)
    cheek_offset_y = int(head_radius * 0.25) # Relative dari hidung
    
    # Gunakan landmark hidung sebagai titik referensi relatif yang stabil
    nose_lm = face.landmark[4]
    cl_pos_x = to_px(nose_lm)[0] - cheek_offset_x
    cr_pos_x = to_px(nose_lm)[0] + cheek_offset_x
    cheek_pos_y = to_px(nose_lm)[1] + cheek_offset_y
    
    cv2.circle(canvas, (cl_pos_x, cheek_pos_y), int(head_radius*0.18), CHEEK_COLOR, -1)
    cv2.circle(canvas, (cr_pos_x, cheek_pos_y), int(head_radius*0.18), CHEEK_COLOR, -1)

    # Mata (Sama, sudah stabil)
    eye_sz = int(head_radius * 0.22)
    def draw_eye_fixed(top_idx, bot_idx, center_idx, iris_idx):
        pos = to_px(face.landmark[center_idx])
        ratio = calc_dist(face.landmark[top_idx], face.landmark[bot_idx]) / calc_dist(eye_l_lm, face.landmark[133])
        if ratio < 0.18:
            cv2.ellipse(canvas, pos, (eye_sz, int(eye_sz*0.4)), angle_tilt, 180, 360, EYE_COLOR, 4, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pos, eye_sz, (255,255,255), -1)
            ip = to_px(face.landmark[iris_idx])
            dx, dy = ip[0]-pos[0], ip[1]-pos[1]
            dx, dy = max(-eye_sz*0.3, min(eye_sz*0.3, dx)), max(-eye_sz*0.3, min(eye_sz*0.3, dy))
            pp = (int(pos[0]+dx), int(pos[1]+dy))
            cv2.circle(canvas, pp, int(eye_sz*0.75), EYE_COLOR, -1)
            cv2.circle(canvas, (pp[0]-int(eye_sz*0.25), pp[1]-int(eye_sz*0.25)), int(eye_sz*0.25), (255,255,255), -1)

    draw_eye_fixed(159, 145, 159, 468); draw_eye_fixed(386, 374, 386, 473)

    # Hidung (TIDAK DIPUTAR LAGI)
    nose_pos = to_px(face.landmark[4])
    nose_sz = int(head_radius * 0.12)
    # Hidung sekarang hanya digambar oval tanpa rotasi sudut
    cv2.ellipse(canvas, (nose_pos[0], nose_pos[1]-nose_sz//2), (nose_sz, nose_sz//2 + 2), 0, 0, 360, NOSE_COLOR, -1)

    # Mulut (TIDAK DIPUTAR LAGI)
    mouth_pos_lm = face.landmark[13] # Landmark bibir atas
    mouth_pos_px = to_px(mouth_pos_lm)
    mouth_open_ratio = calc_dist(face.landmark[13], face.landmark[14]) / calc_dist(face.landmark[10], face.landmark[152])
    
    max_mouth_h = head_radius * 0.4 
    current_mouth_h = mouth_open_ratio * head_radius * 2.5
    final_mouth_h = int(min(current_mouth_h, max_mouth_h))
    
    if mouth_open_ratio > 0.06:
        # Gambar mulut oval tanpa rotasi sudut
        cv2.ellipse(canvas, (mouth_pos_px[0], mouth_pos_px[1] + 5), 
                   (int(head_radius*0.15), final_mouth_h), 
                   0, 0, 360, MOUTH_COLOR, -1)
        
        if final_mouth_h > head_radius * 0.2:
             tongue_y = mouth_pos_px[1] + final_mouth_h - int(final_mouth_h*0.4)
             cv2.circle(canvas, (mouth_pos_px[0], tongue_y), int(head_radius*0.1), (255, 100, 100), -1)
    else:
        mw, mh = int(head_radius * 0.1), int(head_radius * 0.08)
        # Gambar mulut 'w' tanpa rotasi manual
        cv2.ellipse(canvas, (mouth_pos_px[0]-mw//2, mouth_pos_px[1]+10), (mw//2, mh), 0, 0, 180, MOUTH_COLOR, 3, cv2.LINE_AA)
        cv2.ellipse(canvas, (mouth_pos_px[0]+mw//2, mouth_pos_px[1]+10), (mw//2, mh), 0, 0, 180, MOUTH_COLOR, 3, cv2.LINE_AA)

    # 7. TANGAN & PAWS 
    arm_thick_o = int(head_radius*0.35) 
    arm_thick_i = int(head_radius*0.25) 

    def draw_fluffy_hand(hand_landmarks):
        wrist = to_px(hand_landmarks.landmark[0])
        mid_finger = to_px(hand_landmarks.landmark[9])
        palm_center = ((wrist[0]+mid_finger[0])//2, (wrist[1]+mid_finger[1])//2)
        palm_size = int(head_radius * 0.35)
        cv2.circle(canvas, palm_center, palm_size+4, RABBIT_SHADOW, -1)
        finger_tips = [4, 8, 12, 16, 20]
        knuckles = [2, 5, 9, 13, 17]
        finger_thick = int(head_radius * 0.22)
        for i in range(5):
            p_base = to_px(hand_landmarks.landmark[knuckles[i]])
            p_tip = to_px(hand_landmarks.landmark[finger_tips[i]])
            cv2.line(canvas, p_base, p_tip, RABBIT_SHADOW, finger_thick+4, cv2.LINE_AA)
            cv2.line(canvas, p_base, p_tip, RABBIT_FUR, finger_thick, cv2.LINE_AA)
        cv2.circle(canvas, palm_center, palm_size, RABBIT_FUR, -1)
        cv2.circle(canvas, palm_center, int(palm_size*0.6), CHEEK_COLOR, -1)

    # Lengan Kiri
    if pt_shldr_l and pose.landmark[13].visibility > VIS_THRESH:
        pt_elbow_l = to_px(pose.landmark[13])
        # Hitung titik bahu yang disesuaikan
        # Geser pt_shldr_l sedikit ke samping (x) untuk menempel ke tepi badan
        shoulder_offset_x = int(arm_thick_o * 0.8) # Sesuaikan nilai ini jika perlu
        adjusted_shldr_l = (pt_shldr_l[0] - shoulder_offset_x, pt_shldr_l[1]) # Geser ke kiri

        cv2.line(canvas, adjusted_shldr_l, pt_elbow_l, RABBIT_SHADOW, arm_thick_o, cv2.LINE_AA)
        cv2.line(canvas, adjusted_shldr_l, pt_elbow_l, RABBIT_FUR, arm_thick_i, cv2.LINE_AA)
        
        if results.left_hand_landmarks:
            pt_wrist_l = to_px(pose.landmark[15])
            cv2.line(canvas, pt_elbow_l, pt_wrist_l, RABBIT_SHADOW, arm_thick_o, cv2.LINE_AA)
            cv2.line(canvas, pt_elbow_l, pt_wrist_l, RABBIT_FUR, arm_thick_i, cv2.LINE_AA)
            draw_fluffy_hand(results.left_hand_landmarks)
        else:
            cv2.circle(canvas, pt_elbow_l, int(head_radius*0.2), RABBIT_FUR, -1)

    # Lengan Kanan
    if pt_shldr_r and pose.landmark[14].visibility > VIS_THRESH:
        pt_elbow_r = to_px(pose.landmark[14])
        # Hitung titik bahu yang disesuaikan
        # Geser pt_shldr_r sedikit ke samping (x) untuk menempel ke tepi badan
        shoulder_offset_x = int(arm_thick_o * 0.8) # Sesuaikan nilai ini jika perlu
        adjusted_shldr_r = (pt_shldr_r[0] + shoulder_offset_x, pt_shldr_r[1]) # Geser ke kanan

        cv2.line(canvas, adjusted_shldr_r, pt_elbow_r, RABBIT_SHADOW, arm_thick_o, cv2.LINE_AA)
        cv2.line(canvas, adjusted_shldr_r, pt_elbow_r, RABBIT_FUR, arm_thick_i, cv2.LINE_AA)

        if results.right_hand_landmarks:
            pt_wrist_r = to_px(pose.landmark[16])
            cv2.line(canvas, pt_elbow_r, pt_wrist_r, RABBIT_SHADOW, arm_thick_o, cv2.LINE_AA)
            cv2.line(canvas, pt_elbow_r, pt_wrist_r, RABBIT_FUR, arm_thick_i, cv2.LINE_AA)
            draw_fluffy_hand(results.right_hand_landmarks)
        else:
            cv2.circle(canvas, pt_elbow_r, int(head_radius*0.2), RABBIT_FUR, -1)

    # 8. KAKI BARU! (Dari hip ke bawah)
    leg_thick_o = int(head_radius * 0.4) 
    leg_thick_i = int(head_radius * 0.3) 

    if pt_hip_l and pose.landmark[25].visibility > VIS_THRESH:
        pt_knee_l = to_px(pose.landmark[25])
        # Paha
        cv2.line(canvas, pt_hip_l, pt_knee_l, RABBIT_SHADOW, leg_thick_o+4, cv2.LINE_AA)
        cv2.line(canvas, pt_hip_l, pt_knee_l, RABBIT_FUR, leg_thick_i, cv2.LINE_AA)
        
        if pose.landmark[27].visibility > VIS_THRESH: # Ankle
            pt_ankle_l = to_px(pose.landmark[27])
            # Betis
            cv2.line(canvas, pt_knee_l, pt_ankle_l, RABBIT_SHADOW, leg_thick_o, cv2.LINE_AA)
            cv2.line(canvas, pt_knee_l, pt_ankle_l, RABBIT_FUR, leg_thick_i, cv2.LINE_AA)
            
            # Kaki bawah (foot)
            foot_w = int(head_radius * 0.5)
            foot_h = int(head_radius * 0.3)
            # Offset kecil agar terlihat menapak ke depan
            foot_center_x = pt_ankle_l[0] + int(head_radius * 0.1) 
            foot_center_y = pt_ankle_l[1] + int(head_radius * 0.1) 
            
            # Gambar oval kecil untuk tapak kaki
            cv2.ellipse(canvas, (foot_center_x, foot_center_y), (foot_w+4, foot_h+4), 0, 0, 360, RABBIT_SHADOW, -1)
            cv2.ellipse(canvas, (foot_center_x, foot_center_y), (foot_w, foot_h), 0, 0, 360, RABBIT_FUR, -1)
            cv2.circle(canvas, (foot_center_x, foot_center_y), int(foot_w*0.4), CHEEK_COLOR, -1) # Pads

    if pt_hip_r and pose.landmark[26].visibility > VIS_THRESH:
        pt_knee_r = to_px(pose.landmark[26])
        cv2.line(canvas, pt_hip_r, pt_knee_r, RABBIT_SHADOW, leg_thick_o+4, cv2.LINE_AA)
        cv2.line(canvas, pt_hip_r, pt_knee_r, RABBIT_FUR, leg_thick_i, cv2.LINE_AA)
        
        if pose.landmark[28].visibility > VIS_THRESH:
            pt_ankle_r = to_px(pose.landmark[28])
            cv2.line(canvas, pt_knee_r, pt_ankle_r, RABBIT_SHADOW, leg_thick_o, cv2.LINE_AA)
            cv2.line(canvas, pt_knee_r, pt_ankle_r, RABBIT_FUR, leg_thick_i, cv2.LINE_AA)

            foot_w = int(head_radius * 0.5)
            foot_h = int(head_radius * 0.3)
            foot_center_x = pt_ankle_r[0] - int(head_radius * 0.1) # Offset ke kiri untuk kaki kanan
            foot_center_y = pt_ankle_r[1] + int(head_radius * 0.1) 
            
            cv2.ellipse(canvas, (foot_center_x, foot_center_y), (foot_w+4, foot_h+4), 0, 0, 360, RABBIT_SHADOW, -1)
            cv2.ellipse(canvas, (foot_center_x, foot_center_y), (foot_w, foot_h), 0, 0, 360, RABBIT_FUR, -1)
            cv2.circle(canvas, (foot_center_x, foot_center_y), int(foot_w*0.4), CHEEK_COLOR, -1) # Pads

    return canvas

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    model_complexity=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    refine_face_landmarks=True) as holistic:

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # --- PERSIAPAN DRAWING MESH ---
        frame_with_mesh = frame.copy() 

        # --- AKTIFKAN VISUALISASI MESH ---
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame_with_mesh, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                     mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame_with_mesh, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                     mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame_with_mesh, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                     None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_with_mesh, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing_styles.get_default_pose_landmarks_style())
        
        # --- RENDER VTUBER ---
        anim = np.zeros((h, w, 3), dtype=np.uint8)
        anim[:] = BG_NORMAL
        
        if results.face_landmarks and results.pose_landmarks:
            anim = draw_cute_rabbit_v10(anim, results, (h,w,3))
        else:
             cv2.putText(anim, "Mencari Pose...", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow('Real Life (Mesh ON)', frame_with_mesh) 
        cv2.imshow('Rabbit VTuber V10 (Final Body & Stable Face)', anim)

        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()