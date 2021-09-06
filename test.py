import cv2

numbers = cv2.imread("characters/A.jpg")
plate = cv2.imread("contorno.png")
numbers = cv2.resize(numbers, (136, 320))# 136, 320 ORG
ret, bw_img = cv2.threshold(numbers, 127,255, cv2.THRESH_BINARY)
row, col = 140,60

plate[row:row+320, col:col + 136, :] = numbers


#cv2.imshow("number", numbers)
cv2.imshow("plate", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()