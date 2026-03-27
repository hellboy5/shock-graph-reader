import math

def write_esf(filepath, num_samples, point_generator):
    """Helper to format and write the ESF file."""
    with open(filepath, 'w') as f:
        f.write("Extrinsic Shock File v1.0\n\n")
        f.write("Begin [GENERAL INFO]\nNumber of Nodes: 2\nNumber of Edges: 1\nEnd [GENERAL INFO]\n\n")
        f.write("Begin [NODE DESCRIPTION]\n")
        
        # FIXED: Use 'T' instead of 'TERMINAL'
        f.write(f"1 T I [2] [1]\n2 T I [1] [{num_samples}]\n")
        
        f.write("End [NODE DESCRIPTION]\n\n")
        
        f.write("Begin [EDGE DESCRIPTION]\n")
        sample_ids = " ".join(str(i) for i in range(1, num_samples + 1))
        f.write(f"10 I [1 2] [{sample_ids}]\nEnd [EDGE DESCRIPTION]\n\n")
        
        for i in range(num_samples):
            # point_generator yields (x, y, t, theta_deg, speed)
            x, y, t, theta_deg, speed = point_generator(i, num_samples)
            
            f.write("Begin SAMPLE\n")
            f.write(f"sample_id {i+1}\n")
            f.write(f"(x, y, t) ({x:.4f}, {y:.4f}, {t:.4f})\n")
            f.write("edge_id 10\nlabel regular\ntype N\n")
            f.write(f"theta {theta_deg:.4f}\nspeed {speed:.4f}\n")
            f.write("End SAMPLE\n\n")

def gen_arc(i, num_samples):
    """Constant thickness, pure bending (Semicircle)."""
    R = 100.0
    alpha = (i / (num_samples - 1)) * math.pi
    x = R * math.cos(alpha)
    y = R * math.sin(alpha)
    t = 20.0
    theta_deg = math.degrees(alpha + (math.pi / 2.0))
    speed = 99999.0 # Parallel boundaries (phi = 90 deg)
    return x, y, t, theta_deg, speed

def gen_wedge(i, num_samples):
    """Straight spine, pure tapering."""
    L = 100.0
    ratio = i / (num_samples - 1)
    
    x = L * ratio
    y = 0.0
    t = 1.0 + (20.0 * ratio) # Tapers from 1 to 21
    
    dt_ds = 20.0 / L # 0.2
    # In shock math, dt/ds = -cos(phi). So cos(phi) = -0.2
    # Speed = -1 / cos(phi) = -1 / -0.2 = 5.0
    speed = 5.0 
    theta_deg = 0.0 # Straight horizontal line
    
    return x, y, t, theta_deg, speed

def gen_horn(i, num_samples):
    """Curved spine, shrinking thickness (Quarter-circle)."""
    R = 100.0
    ratio = i / (num_samples - 1)
    alpha = ratio * (math.pi / 2.0)
    
    x = R * math.cos(alpha)
    y = R * math.sin(alpha)
    
    # Radius shrinks from 20 to 5
    t = 20.0 - (15.0 * ratio)
    
    # Arc length of quarter circle = 50*pi (~157)
    dt_ds = -15.0 / (50.0 * math.pi)
    cos_phi = -dt_ds
    speed = -1.0 / cos_phi if abs(cos_phi) > 1e-5 else 99999.0
    
    theta_deg = math.degrees(alpha + (math.pi / 2.0))
    
    return x, y, t, theta_deg, speed

if __name__ == "__main__":
    write_esf("synth_arc.esf", 50, gen_arc)
    write_esf("synth_wedge.esf", 50, gen_wedge)
    write_esf("synth_horn.esf", 50, gen_horn)
    print("Successfully generated 3 synthetic test shapes")
