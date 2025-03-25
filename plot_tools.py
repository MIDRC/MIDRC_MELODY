import matplotlib.pyplot as plt
import numpy as np

def plot_spider_chart(groups, values, lower_bounds, upper_bounds, model_name, global_min, global_max, metric=None, plot_config: dict = None):
    # Sort groups so that within each attribute they appear in order
    if plot_config is None:
        plot_config = {}
    custom_orders = plot_config.get('custom_orders', None)
    if custom_orders is None:
        custom_orders = {
            'age_binned': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
            'sex': ['Male', 'Female'],
            'race': ['White', 'Asian', 'Black or African American', 'Other'],
            'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
            'intersectional_race_ethnicity': ['White', 'Not White or Hispanic or Latino'],
        }
    def group_sort_key(label, custom_orders):
        attr, group = label.split(': ', 1)
        if attr in custom_orders:
            attr_order = list(custom_orders.keys()).index(attr)
            order = custom_orders[attr]
            return (attr_order, order.index(group)) if group in order else (attr_order, len(order))
        else:
            return len(custom_orders), group

    combined = list(zip(groups, values, lower_bounds, upper_bounds))
    combined.sort(key=lambda x: group_sort_key(x[0], custom_orders))
    groups, values, lower_bounds, upper_bounds = zip(*combined)

    num_axes = len(groups)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    if plot_config.get('clockwise', False):  # matplotlib default is counter-clockwise
        angles.reverse()
    start_loc = plot_config.get('start', 'right')  # matplotlib default is right
    if start_loc.startswith('t'):
        angle_rot = np.pi / 2
    elif start_loc.startswith('l'):
        angle_rot = np.pi
    elif start_loc.startswith('b'):
        angle_rot = 3 * np.pi / 2
    else:
        angle_rot = 0
    angles = [(angle + angle_rot) % (2 * np.pi) for angle in angles]

    # Close the loop for the plotted series
    values, lower_bounds, upper_bounds = map(lambda x: list(x) + [x[0]], [values, lower_bounds, upper_bounds])
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the main line and markers
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    ax.scatter(angles, values, marker='o', color='b')
    ax.set_ylim(global_min, global_max)

    if metric is not None:
        if metric.upper() == 'QWK':
            # Instead of drawing a line between the vertices for baseline,
            # generate a smooth circle at the 0-level.
            theta_full = np.linspace(0, 2 * np.pi, 100)
            baseline_circle = np.full_like(theta_full, 0)
            ax.plot(theta_full, baseline_circle, color='seagreen', linestyle='--', linewidth=3, alpha=0.8)
        elif metric.upper() == 'EOD':
            ax.fill_between(angles, -0.1, 0.1, color='lightgreen', alpha=0.5)
        elif metric.upper() == 'AAOD':
            ax.fill_between(angles, 0, 0.1, color='lightgreen', alpha=0.5)
            ax.set_ylim(0, global_max)


    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=8, ha='center')
    title = '' if metric is None else f'{metric.upper()} ' + f'Spider Plot for {model_name}'
    ax.set_title(title, size=14, weight='bold')

    if metric.upper() == 'QWK':  # This is a QWK plot
        # Emphasize tick labels based on bounds.
        # Iterate over the tick labels (skipping the last repeated label)
        for i, label in enumerate(ax.get_xticklabels()):
            if i >= len(lower_bounds) - 1:
                break
            angle = angles[i]
            # Calculate rotation so that the dash is perpendicular to the corresponding radius.
            # For angle=0, we want 90 degrees; for angle=pi/2, we want 0 degrees, etc.
            rotation_deg = 90 + np.degrees(angle)
            # If lower bound is greater than zero, mark with blue bold text
            if lower_bounds[i] > 0:
                label.set_fontweight('bold')
                label.set_color('maroon')
                # Add a maroon point at the lower bound for emphasis.
                ax.text(angle, lower_bounds[i], '—', color='maroon', fontsize=12,
                        rotation=rotation_deg, ha='center', va='center')
            # If upper bound is less than zero, mark with red bold text
            elif upper_bounds[i] < 0:
                label.set_fontweight('bold')
                label.set_color('red')
                # Add a red point at the upper bound for emphasis.
                ax.text(angle, upper_bounds[i], '—', color='red', fontsize=12,
                        rotation=rotation_deg, ha='center', va='center')

        plt.tight_layout()
    return fig

def figure_to_image(fig):
    # Draw the renderer
    fig.canvas.draw()
    # Get width and height in pixels
    width, height = fig.canvas.get_width_height()
    # Convert canvas to a numpy array (ARGB)
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(height, width, 4)
    # Convert ARGB to RGB by discarding the alpha channel
    # The buffer order is: [A, R, G, B]
    rgb_img = img[:, :, 1:4]
    return rgb_img


def display_figures_grid(figures, n_cols=3):
    # Calculate the number of rows needed
    n_figs = len(figures)
    n_rows = (n_figs + n_cols - 1) // n_cols

    # Create a figure with subplots for the grid display
    grid_fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8))

    # In case there's a single row, make axes iterable
    if n_rows == 1:
        axes = np.array(axes)
    axes = axes.flatten()

    for i, fig in enumerate(figures):
        img = figure_to_image(fig)
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis for clarity
    # Remove any unused subplots
    for ax in axes[n_figs:]:
        ax.remove()

    plt.tight_layout()
