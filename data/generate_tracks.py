import numpy as np
import csv

def generate_drag_strip(filename, length=200.0, radius=20.0, width=12.0):
    # A long straight going positive X
    # A U-turn at the right end
    # A long straight going negative X
    # A U-turn at the left end
    
    pts = []
    
    # Bottom straight (going right), from x=0 to x=length, y=-radius
    xs = np.linspace(0, length, 20)
    for x in xs:
        pts.append((x, -radius))
        
    # Right U-turn (semi-circle) from -pi/2 to pi/2
    angles = np.linspace(-np.pi/2, np.pi/2, 20)[1:]
    for a in angles:
        pts.append((length + radius * np.cos(a), radius * np.sin(a)))
        
    # Top straight (going left), from x=length to x=0, y=radius
    xs = np.linspace(length, 0, 20)[1:]
    for x in xs:
        pts.append((x, radius))
        
    # Left U-turn (semi-circle) from pi/2 to 3pi/2
    # Don't duplicate the connection points
    angles = np.linspace(np.pi/2, 3*np.pi/2, 20)[1:-1]
    for a in angles:
        pts.append((0 + radius * np.cos(a), radius * np.sin(a)))
        
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
        for p in pts:
            writer.writerow([f"{p[0]:.2f}", f"{p[1]:.2f}", f"{width:.1f}", f"{width:.1f}"])

if __name__ == "__main__":
    generate_drag_strip("tracks/drag_strip.csv", length=300.0, radius=30.0, width=15.0)
    print("Generated tracks/drag_strip.csv")
