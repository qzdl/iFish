import imageio
import numpy as np
from math import sqrt
import sys
import argparse
import os


def get_fish_xn_yn(source_x, source_y, radius, distortion, increment):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    # want to get a curve of m^2, allowing zeros on bounds (edges)
    # can we plot the relationship between (source_x, source_y
    
    
    # return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))
    radius_div = 1  # scaling
    radius_power = 2
    divisor = 1
    
    left_corner_double_center = lambda c: (c*(c**3)) - radius/1-c*radius # double center? left corner is together
    totally_circular = lambda c: (c*(c**3)) - radius/2-radius
    interesting_swirly = lambda c: (1-c**3)-(1-c**4) * radius
    mirrored_inward_radial_fold = lambda c: c / 1 - (radius * c)
    non_mirrored_extreme_radial_edge = lambda c: c / radius - c
    i_think_scaled_down_non_mirrored_extreme_radial_edge = lambda c: c / radius - 2*c
    mirrored_inward_radial_fold_alternate = lambda c: (c / radius - c**4) 
    orig = lambda c: c / (1 - (distortion*(radius**2)))
    orig_param = lambda c: c / (divisor - (distortion*(radius**radius_power)))

    orig = lambda c: (2.5+increment)*c / (1 + ((3 + increment)*(radius)))
    mapper = orig
    return mapper(source_x), mapper(source_y)


def fish(img, distortion_coefficient, increment):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    #print("w,h", w,h, "sqrt(w**2 + h**2)", sqrt(w**2 + h**2)) 
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    edge_threshold = .9
    bound = lambda c: abs(c) < abs(edge_threshold) # FIXME demonstrate threshold
    rds = []
    
    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):
        
            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)
    
            # if not bound(xnd) or not bound(ynd):
            #      dstimg[x][y] = img[x][y]
            #      continue
            
            # get xn and yn distance from normalized center
            rd = sqrt(xnd**2 + ynd**2)
            rds.append(rd)
            if rd > 0.9:
                pass#print('almost fully from edgo', rd)
            
            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient, increment)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Apply fish-eye effect to images.",
        prog='python3 fish.py')

    parser.add_argument("-i", "--image", help="path to image file."
                        " If no input is given, the supplied example 'grid.jpg' will be used.",
                        type=str, default="grid.jpg")

    parser.add_argument("-o", "--outpath", help="file path to write output to."
                        " format: <path>.<format(jpg,png,etc..)>",
                        type=str, default="fish.png")

    parser.add_argument("-d", "--distortion",
                        help="The distoration coefficient. How much the move pixels from/to the center."
                        " Recommended values are between -1 and 1."
                        " The bigger the distortion, the further pixels will be moved outwars from the center (fisheye)."
                        " The Smaller the distortion, the closer pixels will be move inwards toward the center (rectilinear)."
                        " For example, to reverse the fisheye effect with --distoration 0.5,"
                        " You can run with --distortion -0.3."
                        " Note that due to double processing the result will be somewhat distorted.",
                        type=float, default=0.5)

    return parser.parse_args(args)

def make_grid(x, y, interval, width):
    px = [255,255,255,255]
    b_px = [0,0,0,0]
    xy = np.full((10, 4), px)
    grid = np.full((10,10,4), xy)

    for x in range(len(grid)):
        for y in range(len(grid[x])):
           if True in [(x - wI) % interval == 0 or (y - wI) % interval == 0 for wI in range(width + 1)]:
                grid[x][y] = b_px

    return grid

if __name__ == "__main__":
    args = parse_args()
    try:
        imgobj = imageio.imread(args.image)
    except Exception as e:
        print(e)
        sys.exit(1)
    if False and os.path.exists(args.outpath): # FIXME short circuit test 'always overwrite'
        ans = input(
            args.outpath + " exists. File will be overridden. Continue? y/n: ")
        if ans.lower() != 'y':
            print("exiting")
            sys.exit(0)

    output_img = fish(imgobj, args.distortion, 0)
    imageio.imwrite(args.outpath, output_img, format='png')
    # for i in range(30):
    #     print(f"chopi is my sweetest ever in the world baby... {i/10}")
    #     output_img = fish(imgobj, args.distortion, i)
    #     imageio.imwrite(args.outpath+f'-{i}', output_img, format='png')

