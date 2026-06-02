import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


# ------------------------------------------------------------
# Parameters for elliptical canyon shape
# ------------------------------------------------------------
a = 20.0
b = 5.0
k = 3.0
R = 1.0

# Smaller A means the global minimum is less deep
A = 30.0


# ------------------------------------------------------------
# Parameters for very small, overcomable local minimum
# ------------------------------------------------------------
local_min_center = np.array([30.0, -30.0])
local_min_depth = 4.0
local_min_sigma_x = 5.0
local_min_sigma_y = 5.0


# ------------------------------------------------------------
# Tilt of the whole loss landscape
# ------------------------------------------------------------
tilt_x = 0.2
tilt_y = -0.4
tilt_reference = np.array([0.0, 0.0])


# ------------------------------------------------------------
# Loss landscape
# ------------------------------------------------------------
def elliptical_r(x):
    return np.sqrt((x[0] / a) ** 2 + (x[1] / b) ** 2)


def f_base(x):
    r = elliptical_r(x)
    return A / (1 + np.exp(-k * (r - R)))


def local_minimum(x):
    dx = x[0] - local_min_center[0]
    dy = x[1] - local_min_center[1]

    exponent = -(
        dx**2 / (2 * local_min_sigma_x**2)
        + dy**2 / (2 * local_min_sigma_y**2)
    )

    return -local_min_depth * np.exp(exponent)


def tilt(x):
    dx = x[0] - tilt_reference[0]
    dy = x[1] - tilt_reference[1]

    return tilt_x * dx + tilt_y * dy


def f(x):
    return f_base(x) + local_minimum(x) + tilt(x)


# ------------------------------------------------------------
# Gradients
# ------------------------------------------------------------
def grad_f_base(x):
    x1, x2 = x

    denom = np.sqrt((x1 / a) ** 2 + (x2 / b) ** 2)

    if denom == 0:
        return np.array([0.0, 0.0])

    exp_term = np.exp(-k * (denom - R))
    coeff = A * (k * exp_term) / ((1 + exp_term) ** 2 * denom)

    df_dx1 = coeff * (x1 / a**2)
    df_dx2 = coeff * (x2 / b**2)

    return np.array([df_dx1, df_dx2])


def grad_local_minimum(x):
    dx = x[0] - local_min_center[0]
    dy = x[1] - local_min_center[1]

    g = local_minimum(x)

    dg_dx = g * (-dx / local_min_sigma_x**2)
    dg_dy = g * (-dy / local_min_sigma_y**2)

    return np.array([dg_dx, dg_dy])


def grad_tilt(x):
    return np.array([tilt_x, tilt_y])


def grad_f(x):
    return grad_f_base(x) + grad_local_minimum(x) + grad_tilt(x)


# ------------------------------------------------------------
# Optimizers
# ------------------------------------------------------------
class Optimizer:
    def reset(self):
        pass

    def step(self, grad):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def reset(self):
        pass

    def step(self, grad):
        return -self.lr * grad


class Momentum(Optimizer):
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def reset(self):
        self.v = np.zeros(2)

    def step(self, grad):
        self.v = self.momentum * self.v - self.lr * grad
        return self.v


class AdaGrad(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def reset(self):
        self.h = np.zeros(2)

    def step(self, grad):
        self.h += grad**2
        return -self.lr * grad / (np.sqrt(self.h) + 1e-7)


class RMSProp(Optimizer):
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta

    def reset(self):
        self.eg = np.zeros(2)

    def step(self, grad):
        self.eg = self.beta * self.eg + (1 - self.beta) * grad**2
        return -self.lr * grad / (np.sqrt(self.eg) + 1e-7)


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def reset(self):
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, grad):
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return -self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)


optimizer_classes = {
    "SGD": lambda lr: SGD(lr),
    "Momentum": lambda lr: Momentum(lr),
    "AdaGrad": lambda lr: AdaGrad(lr),
    "RMSProp": lambda lr: RMSProp(lr),
    "Adam": lambda lr: Adam(lr),
}


# ------------------------------------------------------------
# Gradient descent path
# ------------------------------------------------------------
def compute_path(alpha, num_steps, optimizer_name):
    # Start in the corner, behind the local minimum
    x = np.array([34.0, -48.0])

    opt = optimizer_classes[optimizer_name](alpha)
    opt.reset()

    path = [x.copy()]

    for _ in range(num_steps - 1):
        grad = grad_f(x)
        update_step = opt.step(grad)

        # Prevent numerical explosions for very aggressive optimizer settings
        if not np.all(np.isfinite(update_step)):
            break

        if np.linalg.norm(update_step) > 20:
            update_step = update_step / np.linalg.norm(update_step) * 20

        x = x + update_step

        if not np.all(np.isfinite(x)):
            break

        path.append(x.copy())

    return np.array(path)


# ------------------------------------------------------------
# Surface data
# ------------------------------------------------------------
X = np.linspace(-50, 50, 170)
Y = np.linspace(-50, 50, 170)
X_grid, Y_grid = np.meshgrid(X, Y)

R_grid = np.sqrt((X_grid / a) ** 2 + (Y_grid / b) ** 2)

Z_base = A / (1 + np.exp(-k * (R_grid - R)))

DX = X_grid - local_min_center[0]
DY = Y_grid - local_min_center[1]

Z_local = -local_min_depth * np.exp(
    -(
        DX**2 / (2 * local_min_sigma_x**2)
        + DY**2 / (2 * local_min_sigma_y**2)
    )
)

Z_tilt = tilt_x * (X_grid - tilt_reference[0]) + tilt_y * (Y_grid - tilt_reference[1])

Z = Z_base + Z_local + Z_tilt


# ------------------------------------------------------------
# Initial values
# ------------------------------------------------------------
init_alpha = 0.06
init_steps = 260
max_steps = 700
init_optimizer = "SGD"

path = compute_path(init_alpha, max_steps, init_optimizer)
fx_path = np.array([f(p) for p in path])


# ------------------------------------------------------------
# Plot setup
# ------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))

# Make space on the left for optimizer selector
plt.subplots_adjust(left=0.22, bottom=0.18)

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X_grid,
    Y_grid,
    Z,
    cmap='viridis',
    alpha=0.65,
    linewidth=0,
    antialiased=True
)

line, = ax.plot([], [], [], 'r-', lw=2)

# Only current red ball, no start point
current_point = ax.scatter([], [], [], color='red', s=70)

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(np.min(Z), np.max(Z))

ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('$\\mathcal{L}(w)$')
ax.set_title("Gradient Descent with Local Minimum")


# ------------------------------------------------------------
# Slider axes
# ------------------------------------------------------------
slider_ax_alpha = plt.axes([0.28, 0.09, 0.62, 0.03])
slider_ax_step = plt.axes([0.28, 0.05, 0.62, 0.03])
slider_ax_azim = plt.axes([0.28, 0.01, 0.28, 0.03])
slider_ax_elev = plt.axes([0.66, 0.01, 0.24, 0.03])

slider_alpha = Slider(
    slider_ax_alpha,
    'Learning Rate',
    0.01,
    0.99,
    valinit=init_alpha,
    valstep=0.01
)

slider_step = Slider(
    slider_ax_step,
    'Step',
    2,
    max_steps,
    valinit=init_steps,
    valstep=1
)

slider_azim = Slider(
    slider_ax_azim,
    'Azimuth',
    0,
    360,
    valinit=209
)

slider_elev = Slider(
    slider_ax_elev,
    'Elevation',
    0,
    90,
    valinit=22.4
)


# ------------------------------------------------------------
# Optimizer selector on the left
# ------------------------------------------------------------
radio_ax = plt.axes([0.02, 0.42, 0.16, 0.25])
radio_optimizer = RadioButtons(
    radio_ax,
    list(optimizer_classes.keys()),
    active=list(optimizer_classes.keys()).index(init_optimizer)
)

radio_ax.set_title("Optimizer")


# ------------------------------------------------------------
# Update function
# ------------------------------------------------------------
def update(val):
    requested_step = int(slider_step.val)
    azim = slider_azim.val
    elev = slider_elev.val
    alpha = slider_alpha.val
    optimizer_name = radio_optimizer.value_selected

    global path, fx_path

    path = compute_path(alpha, max_steps, optimizer_name)
    fx_path = np.array([f(p) for p in path])

    # Make sure the requested step does not exceed the actually computed path
    step = min(requested_step, len(path) - 1)

    visible_path = path[:step + 1]
    visible_fx_path = fx_path[:step + 1]

    line.set_data(visible_path[:, 0], visible_path[:, 1])
    line.set_3d_properties(visible_fx_path)

    # Put red ball exactly at the last point of the visible line
    current_point._offsets3d = (
        [visible_path[-1, 0]],
        [visible_path[-1, 1]],
        [visible_fx_path[-1]]
    )

    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw_idle()


# ------------------------------------------------------------
# Connect widgets
# ------------------------------------------------------------
slider_alpha.on_changed(update)
slider_step.on_changed(update)
slider_azim.on_changed(update)
slider_elev.on_changed(update)
radio_optimizer.on_clicked(update)


# ------------------------------------------------------------
# Initial draw
# ------------------------------------------------------------
update(None)
plt.show()