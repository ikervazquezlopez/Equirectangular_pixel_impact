import numpy as np
import cv2


def replace_mean_element(mean, n, removed_el, added_el):
    return mean - removed_el/n + added_el/n

def replace_mean_element_list(mean, n, removed_els, added_els):
    if len(removed_els) != len(added_els):
        print("WARNING!! Removed elements and added elements are not the same lenght!")
    r_mean = mean
    for i in range(0, len(removed_els)):
        r_mean = replace_mean_element(r_mean, n, removed_els[i], added_els[i])
    return r_mean

def compute_variance(img):
    img = img.astype(np.int32)
    n = img.shape[0] * img.shape[1]
    print(img.dtype, [...])
    mean = np.mean(img)
    xi = np.sum(img)
    xi_2 = np.sum(img*img)
    var = xi_2/n - 2*mean*xi/n + mean*mean
    return var, xi, xi_2

"""
def replace_variance_element(variance, mean, n, _xi_2, removed_el, added_el):
    removed_var = (removed_el - mean) * (removed_el - mean)
    added_var = (added_el - mean) * (added_el - mean)
    print(removed_var/n, added_var/n)
    return variance - removed_var/n + added_var/n

def replace_variance_element_list(variance, mean, n, removed_els, added_els):
    if len(removed_els) != len(added_els):
        print("WARNING!! Removed elements and added elements are not the same lenght!")
    r_var = variance
    for i in range(0, len(removed_els)):
        r_var = replace_variance_element(r_var, mean, n, removed_els[i], added_els[i])
    return r_var

def reduced_covariance(mean_x, mean_y, n, elements_x, elements_y):
    sigma = 0.0
    if len(elements_x) != len(elements_y):
        print("WARNING!! Element list 1 and element list 2 are not the same lenght!")
    for ex, ey in zip(elements_x, elements_y):
        sigma += (ex - mean_x) * (ey - mean_y)
    return sigma / n
"""


img0 = cv2.imread("pixel_impact.png")

Y_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2YCR_CB)
Y_img1 = Y_img0.copy()
Y_img1[0,0] = np.array(140)
Y_img1[0,1] = np.array(140)
Y_img1[0,2] = np.array(140)

Y_img0_mean = np.mean(cv2.split(Y_img0)[0])
Y_img1_mean = np.mean(cv2.split(Y_img1)[0])

Y_img0_var = np.var(cv2.split(Y_img0)[0])
Y_img1_var = np.var(cv2.split(Y_img1)[0])

print("Img0 mean: {}".format(Y_img0_mean))
print("Img1 mean: {}".format(Y_img1_mean))

print("=============")

print("Img0 Y var: {}".format(Y_img0_var))
print("Img1 Y var: {}".format(Y_img1_var))

print("=============")

Y_img, _, _ = cv2.split(Y_img0)
n = Y_img0.shape[0] * Y_img0.shape[1]
#print("Img0 Y var (pixel modified my method): {}".format(replace_variance_element_list(Y_img0_var, Y_img0_mean, n, [Y_img[0,0],Y_img[0,1],Y_img[0,2]], [np.array(140), np.array(140), np.array(140)])))



print("++++++++")
print(np.var(Y_img))
print(compute_variance(Y_img))
