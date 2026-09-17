import numpy as np

# Плоскость фасада Pi: Z=0, камера в Z>0 (перед стеной). Заглубление d: точка при Z=-d.
def project_to_plane(C, P):
    """Пересечение луча C->P с плоскостью Z=0."""
    t = C[2] / (C[2] - P[2])
    return C + t * (P - C)

Cz = 10000.0            # 10 м до стены
d  = 150.0              # заглубление 150 мм

def experiment(Cx, Cy, x_edge, y_edge, label):
    C = np.array([Cx, Cy, Cz])
    outer = np.array([x_edge, y_edge, 0.0])       # наружная кромка: лежит в Pi
    inner = np.array([x_edge, y_edge, -d])        # внутреннее ребро откоса
    o = project_to_plane(C, outer)
    i = project_to_plane(C, inner)
    delta = i[:2] - outer[:2]

    tan_x = (x_edge - Cx) / (Cz + d)
    tan_y = (y_edge - Cy) / (Cz + d)
    tan_full = np.hypot(tan_x, tan_y)

    w_horizontal = abs(i[0] - o[0])               # ширина ВЕРТИКАЛЬНОГО откоса, поперёк ребра
    d_vec = w_horizontal / abs(tan_x)             # векторная (компонентная) формула
    d_sca = w_horizontal / tan_full               # скалярная формула, как в спецификации

    print(f"\n--- {label} ---")
    print(f"  theta_x={np.degrees(np.arctan(tan_x)):6.2f}  theta_y={np.degrees(np.arctan(tan_y)):6.2f}"
          f"  theta_full={np.degrees(np.arctan(tan_full)):6.2f}")
    print(f"  наружная кромка после ректификации: {o[0]:9.2f} (истина {x_edge})  -> сдвиг {o[0]-x_edge:+.3f} мм")
    print(f"  сдвиг внутреннего ребра: dx={delta[0]:+8.2f}  dy={delta[1]:+8.2f}  |d|={np.linalg.norm(delta):7.2f}")
    print(f"  d*tan_full = {d*tan_full:7.2f}   (модуль сдвига совпадает: {np.isclose(np.linalg.norm(delta), d*tan_full, rtol=1e-6)})")
    print(f"  ГЛУБИНА: истина {d:.1f} | векторная {d_vec:7.2f} | скалярная {d_sca:7.2f}"
          f" -> занижение в {d/d_sca:.2f} раза")
    return o, i

# Камера слева и ниже проёма (штатный ракурс дрона: небольшой горизонтальный, большой вертикальный угол)
xL, xR, y = 3500.0, 5000.0, 3000.0

experiment(Cx=3236.0, Cy=-2823.0, x_edge=xR, y_edge=y, label="theta_x=10, theta_y=30 (дальний откос)")
experiment(Cx=4114.0, Cy=-1733.0, x_edge=xR, y_edge=y, label="theta_x=5,  theta_y=25 (дальний откос)")
experiment(Cx=2281.0, Cy=280.0,   x_edge=xR, y_edge=y, label="theta_x=15, theta_y=15 (дальний откос)")
experiment(Cx=3236.0, Cy=3000.0,  x_edge=xR, y_edge=y, label="theta_x=10, theta_y=0  (чисто горизонт.)")

# Окклюзия ближней грани: камера слева (Cx < xL) -> левый откос не виден
print("\n=== Окклюзия ближней (левой) грани, камера слева ===")
C = np.array([1000.0, 3000.0, Cz])
inner_near = np.array([xL, 3000.0, -d])
p = project_to_plane(C, inner_near)
print(f"  наружная кромка xL={xL}; внутреннее ребро проецируется в x={p[0]:.2f}")
print(f"  -> {'ЗА стеной, не наблюдаемо' if p[0] < xL else 'наблюдаемо'}")
