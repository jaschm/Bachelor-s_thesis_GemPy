# kairauksen_paattyminen.py
import matplotlib.patches as patches

def kairaus_paattynyt_maarasyvyyteen(ax, x_center, y_bottom):
    ax.plot([x_center - 0.075, x_center - 0.075], [y_bottom, y_bottom + 0.35], color="black", linewidth=1.2)
    ax.plot([x_center + 0.075, x_center + 0.075], [y_bottom, y_bottom + 0.35], color="black", linewidth=1.2)
    ax.plot([x_center - 0.15, x_center + 0.15], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)

def kairaus_paattynyt_tiiviiseen_maakerrokseen(ax, x_center, y_bottom):
    ax.plot([x_center - 0.10, x_center + 0.10], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)

def kairaus_paattynyt_kiveen_tai_lohkareeseen(ax, x_center, y_bottom):
    ax.plot([x_center - 0.15, x_center + 0.15], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)
    triangle = patches.Polygon([[x_center, y_bottom + 0.3],
                                [x_center - 0.1, y_bottom + 0.5],
                                [x_center + 0.1, y_bottom + 0.5]], closed=True, color="black")
    ax.add_patch(triangle)

def kairaus_paattynyt_kiilautumalla_kivien_tai_lohkareiden_valiin(ax, x_center, y_bottom):
    ax.plot([x_center - 0.15, x_center + 0.15], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)
    left_triangle = patches.Polygon([[x_center - 0.1, y_bottom + 0.3],
                                     [x_center - 0.15, y_bottom + 0.5],
                                     [x_center - 0.05, y_bottom + 0.5]], closed=True, color="black")
    right_triangle = patches.Polygon([[x_center + 0.1, y_bottom + 0.3],
                                      [x_center + 0.05, y_bottom + 0.5],
                                      [x_center + 0.15, y_bottom + 0.5]], closed=True, color="black")
    ax.add_patch(left_triangle); ax.add_patch(right_triangle)

def kairaus_paattynyt_kiveen_lohkareeseen_tai_kallioon(ax, x_center, y_bottom):
    ax.plot([x_center - 0.15, x_center + 0.15], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)
    left_triangle = patches.Polygon([[x_center - 0.1, y_bottom + 0.3],
                                     [x_center - 0.15, y_bottom + 0.5],
                                     [x_center - 0.05, y_bottom + 0.5]], closed=True, color="black")
    ax.add_patch(left_triangle)
    ax.plot([x_center + 0.1, x_center + 0.05], [y_bottom + 0.3, y_bottom + 0.5], color="black", linewidth=1.2)
    ax.plot([x_center + 0.08, x_center + 0.15], [y_bottom + 0.4, y_bottom + 0.5], color="black", linewidth=1.2)

def kairaus_paattynyt_kallioon_varmistettu_kallio(ax, x_center, y_bottom):
    ax.plot([x_center - 0.15, x_center + 0.15], [y_bottom + 0.3, y_bottom + 0.3], color="black", linewidth=1.2)
    ax.plot([x_center - 0.1, x_center - 0.15], [y_bottom + 0.3, y_bottom + 0.5], color="black", linewidth=1.2)
    ax.plot([x_center - 0.12, x_center - 0.05], [y_bottom + 0.4, y_bottom + 0.5], color="black", linewidth=1.2)
    ax.plot([x_center + 0.1, x_center + 0.05], [y_bottom + 0.3, y_bottom + 0.5], color="black", linewidth=1.2)
    ax.plot([x_center + 0.08, x_center + 0.15], [y_bottom + 0.4, y_bottom + 0.5], color="black", linewidth=1.2)
