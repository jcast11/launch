# app.py
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Physics constants (Earth)
# -----------------------------
MU_EARTH = 3.986004418e14      # m^3/s^2  (GM)
R_EARTH  = 6_371_000.0         # m
OMEGA_E  = 7.2921150e-5        # rad/s (Earth rotation rate)

# Simple atmosphere model (very approximate, but useful for "does it decay" demos)
RHO0 = 1.225                   # kg/m^3 at sea level
H0   = 8_500.0                 # m scale height


# -----------------------------
# Helpers
# -----------------------------
def norm(v):
    return float(np.linalg.norm(v))

def atmosphere_density(alt_m):
    # Exponential model (not valid high up, but decent for an educational toggle)
    if alt_m < 0:
        return RHO0
    return RHO0 * math.exp(-alt_m / H0)

def accel_gravity(r_vec):
    r = norm(r_vec)
    return -MU_EARTH * r_vec / (r**3)

def accel_drag(r_vec, v_vec, Cd, A, m, use_drag):
    if not use_drag:
        return np.zeros(2)

    r = norm(r_vec)
    alt = r - R_EARTH
    if alt > 200_000:  # beyond ~200 km this simple model is too wrong; also drag is tiny
        return np.zeros(2)

    rho = atmosphere_density(alt)
    v = norm(v_vec)
    if v == 0:
        return np.zeros(2)

    # Drag force: Fd = 0.5 * rho * v^2 * Cd * A, opposite direction of velocity
    a_mag = 0.5 * rho * v**2 * Cd * A / m
    return -a_mag * (v_vec / v)

def rk4_step(state, dt, Cd, A, m, use_drag):
    # state = [x, y, vx, vy]
    x, y, vx, vy = state

    def f(s):
        x, y, vx, vy = s
        r_vec = np.array([x, y], dtype=float)
        v_vec = np.array([vx, vy], dtype=float)
        a = accel_gravity(r_vec) + accel_drag(r_vec, v_vec, Cd, A, m, use_drag)
        return np.array([vx, vy, a[0], a[1]], dtype=float)

    s = np.array(state, dtype=float)
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return (s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)).tolist()

def orbital_elements_2d(r_vec, v_vec):
    """
    Returns basic orbit quantities for a planar (2D) orbit:
    specific energy, semi-major axis a, eccentricity e,
    perigee rp, apogee ra, period T (if bound).
    """
    r = norm(r_vec)
    v = norm(v_vec)

    # specific angular momentum (scalar in 2D = z-component)
    h = r_vec[0]*v_vec[1] - r_vec[1]*v_vec[0]

    # specific orbital energy
    eps = 0.5 * v*v - MU_EARTH / r

    # semi-major axis
    if abs(eps) < 1e-12:
        a = np.inf
    else:
        a = -MU_EARTH / (2.0 * eps)

    # eccentricity vector magnitude
    # e_vec = (v x h)/mu - r_hat ; in 2D:
    # v x h_z => [ vy*h, -vx*h ]
    r_hat = r_vec / r
    e_vec = np.array([v_vec[1]*h, -v_vec[0]*h], dtype=float) / MU_EARTH - r_hat
    e = norm(e_vec)

    # perigee/apogee distances for conic section if a is finite
    if np.isfinite(a):
        rp = a * (1.0 - e)
        ra = a * (1.0 + e) if e < 1 else np.inf
    else:
        rp, ra = np.nan, np.nan

    # period if bound ellipse
    if np.isfinite(a) and a > 0 and e < 1:
        T = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)
    else:
        T = np.nan

    return {
        "energy_J_per_kg": eps,
        "a_m": a,
        "e": e,
        "h_m2_per_s": h,
        "rp_m": rp,
        "ra_m": ra,
        "period_s": T,
    }

def make_earth_circle(n=400):
    th = np.linspace(0, 2*np.pi, n)
    return R_EARTH*np.cos(th), R_EARTH*np.sin(th)

def format_km(x_m):
    return f"{x_m/1000:,.1f} km"

def format_speed(x):
    return f"{x:,.1f} m/s"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Satellite Orbit Simulator", layout="wide")
st.title("🛰️ Satellite Launch-to-Orbit Simulator (Realistic Two-Body Physics)")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Initial conditions")

    mode = st.radio(
        "Initialization mode",
        ["Instantaneous injection (start in space)", "From Earth surface (include Earth rotation only)"],
        index=0
    )

    if mode == "Instantaneous injection (start in space)":
        alt_km = st.slider("Initial altitude above Earth (km)", 120, 2000, 300, 10)
        speed = st.slider("Initial speed (m/s)", 5000, 12000, 7800, 50)
        fpa_deg = st.slider("Flight-path angle γ (deg)  (0 = horizontal)", -30.0, 30.0, 0.0, 0.5)
        direction = st.selectbox("Direction", ["Prograde (counterclockwise)", "Retrograde (clockwise)"], index=0)

        r0 = R_EARTH + alt_km*1000.0
        # Start on +x axis; velocity mostly +y for prograde
        x0, y0 = r0, 0.0

        gamma = math.radians(fpa_deg)
        v_tan = speed * math.cos(gamma)
        v_rad = speed * math.sin(gamma)

        # Radial direction at +x is +x, tangential is +y (prograde)
        if direction.startswith("Prograde"):
            vx0 = v_rad
            vy0 = v_tan
        else:
            vx0 = v_rad
            vy0 = -v_tan

    else:
        st.caption("This mode does **not** model thrust/ascent through atmosphere. It starts at the surface and lets you see if the initial velocity leads to orbit or impact.")
        lat_deg = st.slider("Launch latitude (deg)", -90.0, 90.0, 0.0, 0.5)
        speed = st.slider("Initial speed relative to ground (m/s)", 0, 12000, 8000, 50)
        fpa_deg = st.slider("Flight-path angle γ (deg)  (0 = horizontal)", -30.0, 30.0, 0.0, 0.5)
        heading = st.selectbox("Heading", ["Eastward (prograde)", "Westward (retrograde)"], index=0)

        lat = math.radians(lat_deg)
        r0 = R_EARTH
        x0, y0 = r0, 0.0  # local position in our 2D plane

        # Earth rotation contributes tangential velocity: omega*R*cos(lat)
        v_rot = OMEGA_E * R_EARTH * math.cos(lat)
        gamma = math.radians(fpa_deg)
        v_tan_ground = speed * math.cos(gamma)
        v_rad = speed * math.sin(gamma)

        # Tangential direction at +x is +y (eastward is prograde)
        if heading.startswith("Eastward"):
            v_tan_inertial = v_tan_ground + v_rot
            vx0 = v_rad
            vy0 = v_tan_inertial
        else:
            v_tan_inertial = -v_tan_ground + v_rot  # westward ground speed but rotation still eastward
            vx0 = v_rad
            vy0 = v_tan_inertial

    st.divider()
    st.subheader("Simulation controls")

    dt = st.slider("Time step dt (s)", 1, 60, 10, 1)
    sim_hours = st.slider("Simulate duration (hours)", 0.25, 24.0, 6.0, 0.25)
    stop_on_impact = st.checkbox("Stop if it hits Earth", value=True)

    st.divider()
    st.subheader("Optional drag (rough)")

    use_drag = st.checkbox("Enable atmospheric drag", value=False)
    Cd = st.slider("Drag coefficient Cd", 1.0, 3.0, 2.2, 0.1, disabled=not use_drag)
    A = st.slider("Cross-sectional area A (m²)", 0.01, 50.0, 2.0, 0.01, disabled=not use_drag)
    m = st.slider("Mass m (kg)", 1.0, 20_000.0, 500.0, 1.0, disabled=not use_drag)

with right:
    st.subheader("Results")

    r0_vec = np.array([x0, y0], dtype=float)
    v0_vec = np.array([vx0, vy0], dtype=float)

    els0 = orbital_elements_2d(r0_vec, v0_vec)

    alt0 = norm(r0_vec) - R_EARTH
    st.write(
        f"**Initial altitude:** {format_km(alt0)}  \n"
        f"**Initial speed:** {format_speed(norm(v0_vec))}"
    )

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Eccentricity e", f"{els0['e']:.4f}")
    with colB:
        a = els0["a_m"]
        st.metric("Semi-major axis a", "unbound" if (not np.isfinite(a) or a <= 0) else format_km(a))
    with colC:
        rp = els0["rp_m"]
        st.metric("Perigee altitude", "—" if not np.isfinite(rp) else format_km(rp - R_EARTH))

    rp = els0["rp_m"]
    ra = els0["ra_m"]
    colD, colE, colF = st.columns(3)
    with colD:
        st.metric("Apogee altitude", "—" if not np.isfinite(ra) else ("unbound" if np.isinf(ra) else format_km(ra - R_EARTH)))
    with colE:
        T = els0["period_s"]
        st.metric("Period", "—" if (not np.isfinite(T)) else f"{T/60:,.1f} min")
    with colF:
        eps = els0["energy_J_per_kg"]
        st.metric("Specific energy", f"{eps:,.0f} J/kg")

    st.divider()

    # -----------------------------
    # Run simulation
    # -----------------------------
    n_steps = int(sim_hours * 3600 / dt)
    state = [x0, y0, vx0, vy0]

    xs, ys = [], []
    vmag = []
    alt = []
    t_arr = []

    impacted = False
    impact_step = None

    for i in range(n_steps + 1):
        x, y, vx, vy = state
        r = math.hypot(x, y)

        xs.append(x)
        ys.append(y)
        vmag.append(math.hypot(vx, vy))
        alt.append(r - R_EARTH)
        t_arr.append(i * dt)

        if r <= R_EARTH:
            impacted = True
            impact_step = i
            if stop_on_impact:
                break

        state = rk4_step(state, dt, Cd, A, m, use_drag)

    xs = np.array(xs); ys = np.array(ys)
    alt = np.array(alt); vmag = np.array(vmag); t_arr = np.array(t_arr)

    if impacted:
        st.warning(f"Impact detected at t ≈ {t_arr[-1]/60:,.1f} minutes.", icon="⚠️")

    # Orbit plot
    ex, ey = make_earth_circle()
    fig_orbit = go.Figure()
    fig_orbit.add_trace(go.Scatter(x=ex/1000, y=ey/1000, mode="lines", name="Earth"))
    fig_orbit.add_trace(go.Scatter(x=xs/1000, y=ys/1000, mode="lines", name="Trajectory"))
    fig_orbit.add_trace(go.Scatter(x=[xs[0]/1000], y=[ys[0]/1000], mode="markers", name="Start"))

    fig_orbit.update_layout(
        height=520,
        xaxis_title="x (km)",
        yaxis_title="y (km)",
        yaxis_scaleanchor="x",
        legend_orientation="h",
        legend_y=-0.15,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig_orbit, use_container_width=True)

    # Altitude/time plot
    fig_alt = go.Figure()
    fig_alt.add_trace(go.Scatter(x=t_arr/60, y=alt/1000, mode="lines", name="Altitude"))
    fig_alt.update_layout(
        height=260,
        xaxis_title="Time (min)",
        yaxis_title="Altitude (km)",
        margin=dict(l=10, r=10, t=10, b=10)
    )

    # Speed/time plot
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t_arr/60, y=vmag, mode="lines", name="Speed"))
    fig_v.update_layout(
        height=260,
        xaxis_title="Time (min)",
        yaxis_title="Speed (m/s)",
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig_alt, use_container_width=True)
    st.plotly_chart(fig_v, use_container_width=True)

st.caption(
    "Model: planar two-body dynamics with optional simple atmospheric drag. "
    "For classroom use this is a strong ‘real physics’ baseline; adding thrust, staging, and full 3D inclination is possible as an extension."
)
