import os
import cv2

root_path = '/media/edutech-pc06/Elements1/DataSet/ClasificacionPorContenido/PhotoChartDigital/Chart/complex_charts'
files = os.listdir(root_path)
for file in files:
    img = cv2.imread(root_path+'/'+file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is not None:
        # img = cv2.equalizeHist(img)
        # gauss = cv2.GaussianBlur(img, (5, 5), 0)
        # canny = cv2.Canny(gauss, 50, 150)
        #
        # cont, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        #
        # cv2.drawContours(img, cont, -1, (0, 0, 255), 2)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
        cv2.imshow(file, img)
        cv2.waitKey()
        cv2.destroyWindow(file)
