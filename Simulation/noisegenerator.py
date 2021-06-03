import numpy as np

# shifts the original signal in a selected direction
def moveNoise(noise_dir, qnt):
    x1 = 0
    y1 = 0
    if 'bottom' in noise_dir:
        x1 = qnt
    elif 'top' in noise_dir:
        x1 = -qnt
    if 'right' in noise_dir:
        y1 = qnt
    elif 'left'  in noise_dir:
        y1 = -qnt
    return x1, y1
 
# generates a certain amount of noise to the original signal   
def generateNoise(period, noise_qnt, noise_dir):
    noise_x = 0
    noise_y = 0
    if "2" in noise_qnt:
        if noise_qnt == "low2medium" or noise_qnt == "medium2low":
            x21 = 0.02
            y21 = 0.02
            x22 = 0.06
            y22 = 0.05
            x11, y11 = moveNoise(noise_dir, 0.1)
            x12, y12 = moveNoise(noise_dir, 0.4)
        elif noise_qnt == "medium2hight" or noise_qnt == "hight2medium":
            x21 = 0.06
            y21 = 0.05
            x22 = 0.2
            y22 = 0.1
            x11, y11 = moveNoise(noise_dir, 0.4)
            x12, y12 = moveNoise(noise_dir, 0.6)
            
        if noise_qnt == "low2medium" or noise_qnt == "medium2hight":
            noise_x = np.concatenate((np.random.normal(x11, x21, period-round(period/2)), np.random.normal(x12, x22, round(period/2))))
            noise_y = np.concatenate((np.random.normal(y11, y21, period-round(period/2)), np.random.normal(y12, y22, round(period/2))))
        else:
            noise_x = np.concatenate((np.random.normal(x12, x22, round(period/2)), np.random.normal(x11, x21, period-round(period/2))))
            noise_y = np.concatenate((np.random.normal(y12, y22, round(period/2)), np.random.normal(y11, y21, period-round(period/2))))
    elif noise_qnt != '':
        if noise_qnt == 'low' or noise_qnt == 'lowR':
            x2 = 0.02
            y2 = 0.02
            x1, y1 = moveNoise(noise_dir, 0.1)
        elif noise_qnt == 'medium' or noise_qnt == 'mediumR':
            x2 = 0.06
            y2 = 0.05
            x1, y1 = moveNoise(noise_dir, 0.4)
        elif noise_qnt == 'hight' or noise_qnt == 'hightR':
            x2 = 0.2
            y2 = 0.1
            x1, y1 = moveNoise(noise_dir, 0.6)
                
        noise_x = np.random.normal(x1, x2, period)
        noise_y = np.random.normal(y1, y2, period)
    return noise_x, noise_y

# generates oblique noise
def generateObliqueDistortion(noise_dir, noise_qnt):
    from_pos_noise, to_pos_noise = 0, 0
    if noise_dir == 'left':
        if noise_qnt == 'low2medium':
            from_pos_noise = 0
            to_pos_noise = -0.2
        elif noise_qnt == 'medium2low':
            from_pos_noise = -0.2
            to_pos_noise = 0
        elif noise_qnt == 'medium2hight':
            from_pos_noise = 0.3
            to_pos_noise = -0.4
        elif noise_qnt == 'hight2medium':
            from_pos_noise = -0.4
            to_pos_noise = 0.3
    elif noise_dir == 'right':
        if noise_qnt == 'low2medium':
            from_pos_noise = -0.2
            to_pos_noise = 0
        elif noise_qnt == 'medium2low':
            from_pos_noise = 0
            to_pos_noise = -0.2
        elif noise_qnt == 'medium2hight':
            from_pos_noise = -0.4
            to_pos_noise = 0.3
        elif noise_qnt == 'hight2medium':
            from_pos_noise = 0.3
            to_pos_noise = -0.4
    elif noise_dir == 'top-left':
        if noise_qnt == 'low2medium' or noise_qnt == 'low':
            from_pos_noise = 0
            to_pos_noise = -0.2
        elif noise_qnt == 'medium2low' or noise_qnt == 'lowR':
            from_pos_noise = -0.2
            to_pos_noise = 0
        elif noise_qnt == 'medium2hight' or noise_qnt == 'medium' or noise_qnt == 'hight':
            from_pos_noise = 0.3
            to_pos_noise = -0.4
        elif noise_qnt == 'hight2medium' or noise_qnt == 'mediumR' or noise_qnt == 'hightR':
            from_pos_noise = -0.4
            to_pos_noise = 0.3
    elif noise_dir == 'top-right':
        if noise_qnt == 'low2medium' or noise_qnt == 'low':
            from_pos_noise = -0.2
            to_pos_noise = 0
        elif noise_qnt == 'medium2low' or noise_qnt == 'lowR':
            from_pos_noise = 0
            to_pos_noise = -0.2
        elif noise_qnt == 'medium2hight' or noise_qnt == 'medium' or noise_qnt == 'hight':
            from_pos_noise = -0.4
            to_pos_noise = 0.3 
        elif noise_qnt == 'hight2medium' or noise_qnt == 'mediumR' or noise_qnt == 'hightR':
            from_pos_noise = 0.3
            to_pos_noise = -0.4
    
    return from_pos_noise, to_pos_noise