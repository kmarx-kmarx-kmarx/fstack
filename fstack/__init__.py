import numpy as np
import cv2
import scipy.ndimage
import time

def fstack(paths, focus, WSize=9, alpha=0.2, sth=13):
    t0 = time.time()
    print("FMeasure", end = ": ")
    imgs = np.array([cv2.imread(path) for path in paths], dtype='f')
    isColor, stack, h, w, c = parseInputs(imgs)
    
    # If isColor, store the color version and overwrite imgs with black and white
    if(isColor):
        imgs_color = np.copy(imgs)
        imgs = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in imgs], dtype='f')
    # Take only one channel
    if(c>1):
        imgs = imgs[:,:,:,0]
    
    focus_measure = np.array([gfocus(img, WSize) for img in imgs])
    t1 = time.time()
    print(t1 - t0)
    
    print("SMeasure")
    u, s, A, FMax = gauss3P(focus, focus_measure)
    t2 = time.time()
    print("gauss3P: " + str(t2 - t1))
    
    # Estimate RMS error along slice axis
    err = np.array([np.abs(focus_measure[i,:,:] - A * np.exp(-(1+focus[i]-u)**2/(2*(s**2)))) for i in range(len(focus))])
    err[np.isnan(err)] = np.max(err[~np.isnan(err)])
    # Sum along slice axis
    err = np.sum(err, axis=0)
    err = err/(FMax * stack)
    t3 = time.time()
    print("err: " + str(t3 - t2))
    # might need to transpose focus_measure to slice across slices
    # renormalize focus_measure
    focus_measure = [fmeas/FMax for fmeas in focus_measure]
    # Filter the err
    kernel = np.ones((WSize, WSize))/(WSize * WSize)
    inv_psnr = np.array(scipy.ndimage.correlate(err, kernel, mode='nearest'), dtype='f')
        
    S = 20*np.log10(1.0/inv_psnr)
    S[np.isnan(S)] = np.min(S[~np.isnan(S)])
    
    phi = 0.5*(1+np.tanh(alpha*(S-sth)))/alpha
    phi = cv2.medianBlur(phi, 3)
    
    focus_measure = [0.5 + 0.5*np.tanh(phi*(slc-1)) for slc in focus_measure]
    
    # Sum along slice axis
    fmn = np.sum(focus_measure,0)
    
    t4 = time.time()
    print("filter: " + str(t4 - t3))    
    
    if(isColor):
        imgs_color[:,:,0] = np.sum(imgs_color[:,:,0] * focus_measure , axis=0)/fmn
        imgs_color[:,:,1] = np.sum(imgs_color[:,:,1] * focus_measure , axis=0)/fmn
        imgs_color[:,:,2] = np.sum(imgs_color[:,:,2] * focus_measure , axis=0)/fmn
        retval =  imgs_color
    else:
        imgs = np.sum(imgs * focus_measure , axis=0)/fmn
        retval =  imgs
        
    t5 = time.time()
    print("fusion: " + str(t5 - t4))
    return retval
        
def gfocus(im, WSize):
    # verified - matches matlab output
    img = np.array(im/255, dtype='d')
    kernel = np.ones((WSize, WSize))/(WSize * WSize)
    filtered = np.array(scipy.ndimage.correlate(img, kernel, mode='nearest'), dtype='d')
    err = np.square(img-filtered)
    err = np.array(cv2.filter2D(err, -1, kernel), dtype = 'd')
    return err

def gauss3P(x, Y):
    STEP = 2 # Internal parameter
    P,M,N = Y.shape
    Ymax, I = np.max(Y, axis=0), np.argmax(Y, axis=0)# make sure we stay in range
    Ic = np.copy(I)
    Ic[Ic <= STEP] = STEP
    Ic[Ic >= P-STEP-1] = P-STEP-1

    Index1 = Ic - STEP
    Index2 = Ic
    Index3 = Ic + STEP

    Index1[I <= STEP-1] = Index3[I <= STEP-1]
    Index3[I >= STEP-1] = Index1[I >= STEP-1]

    x1 = np.array(Ic - STEP) + 1
    x2 = Ic + 1
    x3 = Ic + STEP + 1
    if x != list(range(len(x))):
        # convert from order index to focus value
        x1 = np.array([[x[x1[i,j]] for j in range(M)] for i in range(N)], dtype='d')
        x2 = np.array([[x[x2[i,j]] for j in range(M)] for i in range(N)], dtype='d')
        x3 = np.array([[x[x3[i,j]] for j in range(M)] for i in range(N)], dtype='d')

    # Index through the image stack and take the elementwise log
    M_IDX, N_IDX = np.indices(Y.shape[1:])
    y1 = np.array(Y[Index1, M_IDX, N_IDX], dtype = 'D')
    y1 = np.log(y1)
    y2 = np.array(Y[Index2, M_IDX, N_IDX], dtype = 'D')
    y2 = np.log(y2)
    y3 = np.array(Y[Index3, M_IDX, N_IDX], dtype = 'D')
    y3 = np.log(y3)

    c = np.array(( (y1-y2)*(x2-x3)-(y2-y3)*(x1-x2) )/( (x1**2-x2**2)*(x2-x3)-(x2**2-x3**2)*(x1-x2) ), dtype = 'D')
    b = ( (y2-y3)-c*(x2-x3)*(x2+x3) )/(x2-x3)
    s = np.sqrt(-1/(2*c))
    u = b*s**2
    a = y1 - b*x1 - c*x1**2
    A = np.exp(a + u**2/(2*s**2))
    
    return np.real(u), np.real(s), np.real(A), np.real(Ymax)


def parseInputs(image_list):
    '''
    Reads and returns properties of the image stack
    '''
    # first read one image to tell whether it is color or black and white
    img = image_list[0]
    isColor = True
    if (len(img.shape) < 3):
        isColor = False
        print("not 3D")
    if(len(img.shape) == 3):
        print("is 3D")
        b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
        if ((b==g).all()) and ((b==r).all()):
            print("all same")
            isColor = False
    
    # Find number of images and size of each image
    if(len(image_list.shape)==4):
        stack, h, w, c = image_list.shape
    else:
        c = 1
        stack, h, w = image_list.shape
        
    return isColor, stack, h, w, c