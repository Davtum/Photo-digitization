import numpy as np

def Kmat(f, cx=2640., cy=1978.):
    return np.array([[f,0,cx],[0,f,cy],[0,0,1.]])

def rot_x(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rectify_aspect(f_true, eps, tilt_deg, pan_deg):
    """Квадрат на плоскости фасада -> изображение (истинное f) -> ректификация с ошибочным f."""
    f  = f_true
    fp = f_true * (1 + eps)
    K, Kp = Kmat(f), Kmat(fp)

    # Ориентация плоскости относительно камеры
    R = rot_y(np.radians(pan_deg)) @ rot_x(np.radians(tilt_deg))
    # Направления осей плоскости в системе камеры
    d1, d2 = R @ np.array([1.,0,0]), R @ np.array([0,1.,0])
    n = np.cross(d1, d2)
    C_to_plane = 10000.0

    # Квадрат 1500x1500 мм на плоскости
    S = 1500.
    corners_plane = np.array([[0,0],[S,0],[S,S],[0,S]], float)
    origin = -n * C_to_plane * np.sign(np.dot(n, np.array([0,0,1.])))
    pts3d = np.array([origin + c[0]*d1 + c[1]*d2 for c in corners_plane])
    if pts3d[:,2].min() <= 0:
        pts3d = np.array([-origin + c[0]*d1 + c[1]*d2 for c in corners_plane])

    img = (K @ pts3d.T).T
    img = img[:,:2] / img[:,2:3]

    # Точки схода направлений плоскости (истинные, из изображения)
    v1 = K @ d1
    v2 = K @ d2

    # Оценка с ошибочным фокусным
    e1 = np.linalg.inv(Kp) @ v1;  e1 /= np.linalg.norm(e1)
    e2 = np.linalg.inv(Kp) @ v2;  e2 /= np.linalg.norm(e2)
    # Грам-Шмидт: направления перестали быть ортогональными
    non_orth = np.degrees(np.arccos(abs(np.clip(np.dot(e1,e2),-1,1))))
    e2o = e2 - np.dot(e2,e1)*e1;  e2o /= np.linalg.norm(e2o)
    e3 = np.cross(e1,e2o)
    Rp = np.column_stack([e1,e2o,e3])

    # Ректификация: луч -> система плоскости -> фронто-параллельные координаты
    rect = []
    for p in img:
        ray = np.linalg.inv(Kp) @ np.array([p[0],p[1],1.])
        q = Rp.T @ ray
        rect.append([q[0]/q[2], q[1]/q[2]])
    rect = np.array(rect)

    w = (np.linalg.norm(rect[1]-rect[0]) + np.linalg.norm(rect[2]-rect[3]))/2
    h = (np.linalg.norm(rect[3]-rect[0]) + np.linalg.norm(rect[2]-rect[1]))/2
    return abs(w/h - 1.0), non_orth

f_true = 3600.
print(f"{'tilt':>6} {'pan':>5} | {'иск-е факт':>11} | {'e*sin^2':>9} | {'e*sin^2/cos':>12} | неортог.")
print("-"*72)
for tilt, pan in [(10,0),(15,0),(20,0),(30,0),(45,0),(30,10),(20,20),(35,15)]:
    for eps in [0.02]:
        err, no = rectify_aspect(f_true, eps, tilt, pan)
        th = np.radians(np.degrees(np.arccos(np.cos(np.radians(tilt))*np.cos(np.radians(pan)))))
        a = eps*np.sin(th)**2
        b = eps*np.sin(th)**2/np.cos(th)
        print(f"{tilt:>6} {pan:>5} | {err*100:>10.3f}% | {a*100:>8.3f}% | {b*100:>11.3f}% | {no:>6.3f}°")
