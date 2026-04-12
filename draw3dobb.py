from __future__ import print_function, division
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D

def tryPlot():
    cmap = plt.get_cmap('jet_r')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Fuselage (nose to tail)
    draw(ax, [0, 0, 0, 1.0, 0, 0, 0.1, 0.1, 0.1, 0.1], cmap(float(1)/7))
    # Right Wing (root to tip)
    draw(ax, [0.4, 0.05, 0, 0.6, 0.5, 0.05, 0.2, 0.02, 0.1, 0.01], cmap(float(2)/7))
    plt.show()

def draw(ax, p, color, label=None):
    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt

    # Determine semantic color if 13D
    if len(p) >= 13:
        cls_idx = np.argmax(p[10:13])
        if cls_idx == 0:
            color = 'gray' # Fuselage
        elif cls_idx == 1:
            color = 'blue' # Wings / Stabilizers
        elif cls_idx == 2:
            color = 'red'  # Engines

    # 10D geometry format: [x1, y1, z1, x2, y2, z2, L1, H1, L2, H2]
    c1 = np.array(p[0:3])
    c2 = np.array(p[3:6])
    L1, H1 = p[6], p[7]
    L2, H2 = p[8], p[9]
    
    # Render label if provided (at the center of the box)
    if label is not None:
        center = (c1 + c2) / 2
        ax.text(center[0], center[1], center[2], str(label), 
                color='black', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # The direction connecting the two opposite face centers (e.g., spanwise for wing)
    dir_span = c2 - c1
    span_len = LA.norm(dir_span)
    
    if span_len < 1e-6:
        # Fallback if points are too close
        dir_span_norm = np.array([0, 1.0, 0])
    else:
        dir_span_norm = dir_span / span_len
        
    # The 'L' dimension is always parallel to the global XY plane if the span is horizontal.
    # However, if the span is primarily vertical (like a vertical stabilizer), 'L' should be 
    # parallel to the global XZ plane (roughly aligned with X). 
    # We choose the reference vector based on the dominant axis of the span.
    if abs(dir_span_norm[2]) > 0.707: # Z is dominant
        # Use -Y so that crossing with +Z yields +X
        ref_vec = np.array([0.0, -1.0, 0.0])
    else:
        ref_vec = np.array([0.0, 0.0, 1.0])
    
    dir_chord = np.cross(dir_span_norm, ref_vec)
    chord_len = LA.norm(dir_chord)
    
    if chord_len < 1e-6:
        dir_chord_norm = np.array([1.0, 0, 0])
    else:
        dir_chord_norm = dir_chord / chord_len
        
    # The 'H' dimension (thickness) direction is orthogonal to both span and chord
    dir_thick_norm = np.cross(dir_chord_norm, dir_span_norm)
    dir_thick_norm = dir_thick_norm / LA.norm(dir_thick_norm)
    
    # Calculate corners for face 1
    d1_chord = 0.5 * L1 * dir_chord_norm
    d1_thick = 0.5 * H1 * dir_thick_norm
    
    # Calculate corners for face 2
    d2_chord = 0.5 * L2 * dir_chord_norm
    d2_thick = 0.5 * H2 * dir_thick_norm
    
    cornerpoints = np.zeros([8, 3])
    
    # Face 1 (at c1)
    cornerpoints[0][:] = c1 - d1_chord - d1_thick
    cornerpoints[1][:] = c1 + d1_chord - d1_thick
    cornerpoints[2][:] = c1 - d1_chord + d1_thick
    cornerpoints[3][:] = c1 + d1_chord + d1_thick
    
    # Face 2 (at c2)
    cornerpoints[4][:] = c2 - d2_chord - d2_thick
    cornerpoints[5][:] = c2 + d2_chord - d2_thick
    cornerpoints[6][:] = c2 - d2_chord + d2_thick
    cornerpoints[7][:] = c2 + d2_chord + d2_thick

    # Edges Face 1
    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]], [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]], [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]], [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]], [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    
    # Edges Face 2
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]], [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]], [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]], [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]], [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    
    # Connecting Edges
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]], [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]], [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]], [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]], [cornerpoints[3][2], cornerpoints[7][2]], c=color)

def showGenshapes(genshapes, labels=None):
    for i in range(len(genshapes)):
        recover_boxes = genshapes[i]
        curr_labels = labels[i] if labels is not None else None

        fig = plt.figure(i)
        cmap = plt.get_cmap('jet_r')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        for jj in range(len(recover_boxes)):
            p = recover_boxes[jj][:]
            label = curr_labels[jj] if curr_labels is not None else None
            draw(ax, p, cmap(float(jj)/len(recover_boxes)), label=label)

        plt.show()

def showGenshape(genshape, labels=None):
    recover_boxes = genshape

    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    for jj in range(len(recover_boxes)):
        p = recover_boxes[jj][:]
        label = labels[jj] if labels is not None else None
        draw(ax, p, cmap(float(jj)/len(recover_boxes)), label=label)

    plt.show()
