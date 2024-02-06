import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('E:\\mymode')
import  cnts_module
import imutils 



img=cv2.imread(r"E:\img\butterfly\card\mode.png")
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('mode',img)
#轉換灰階圖
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
#轉換為二值
ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)
#輪廓檢測
contours,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print(len(contours))
#繪製輪廓線
draw=cv2.drawContours(img,contours,-1,(0,0,255),2)
cv_show('draw',draw)
#排序
refCnts = imutils.contours.sort_contours(contours,method = 'lefy-to-right')[0]
digits={}
for (i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h,x:x+w]
    cv_show('a', roi)
    roi = cv2.resize(roi,(57,88))
    digits[i]=roi
    
#對輸出圖像做處理
#去除噪音項錢處理,指定盒的大小,排除不需要的噪音.
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,2))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,3))
imge = cv2.imread(r"E:\img\butterfly\card\Lifestyle.jpg")
cv_show('imge', imge)
#變更大小
imge = cnts_module.resize(imge,width=300)
cv_show('imge', imge)
#轉灰階
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
cv_show('imge', gray)
#頂帽凸出字體的位置
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT, rectKernel)
cv_show('top', tophat)
#用斯皮爾算子,抓取邊緣
sobelx=cv2.Sobel(tophat, ddepth = cv2.CV_32F, dx=1, dy=0,ksize=-1)
sobelx=cv2.convertScaleAbs(sobelx)
cv_show('sobelxy', sobelx)
# sobely=cv2.Sobel(tophat, ddepth = cv2.CV_32F, dx=0, dy=1,ksize=-1)
# sobely=cv2.convertScaleAbs(sobely)
# sobelxy=cv2.addWeighted(sobelx, 0.9, sobely, 0.2, 0)
# cv_show('sobelxy', sobelxy)
#規一化
(minVal, maxVal) = (np.min(sobelx), np.max(sobelx))
gradX = (255 * ((sobelx - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show('sobelxy', gradX )
#將數字連一起,使用閉運算
CLOSE=cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('close', CLOSE)
#二值化
thresh = cv2.threshold(CLOSE, 0, 255, cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
#填補二值化後的缺口,進行二次閉運算
CLOSE2 = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel )

cv_show('CLOSE2', CLOSE2)
#對閉運算結果的輪廓計算
contour,hierarchy = cv2.findContours(CLOSE2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#繪製輪廓線
draw2 = cv2.drawContours(imge.copy(),contour,-1,(0,0,255),3)
cv_show('draw',draw2)
locs = []
#不重要的特徵進行過濾

for i , c in enumerate (contour):
    #創造輪廓的外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    
    #計算輪廓的長寬比
    ar = w/h
  
    #通過長寬比,進行判斷
    if ar > 0.01 and ar < 5.0:
        if (w >= 30 and w <= 65) and (h >= 16 and h <= 35):
            locs.append((x,y,w,h))
#對輪廓進行排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 枚舉輪廓中的數字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []

	# 根據座標提取每一个组
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	cv_show('group',group)
	# 預處理
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)
	# 計算每一組的輪廓
	digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts =imutils.contours.sort_contours(digitCnts,
		method="left-to-right")[0]

	# 計算每一组中的每一個數值
	for c in digitCnts:
		# 找到當前輪廓大小，resize成合適的的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)

		# 計算匹配得分
		scores = []

		# 在模板中計算每一得分
		for (digit, digitROI) in digits.items():
			# 模板匹配
			result = cv2.matchTemplate(roi, digitROI,
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# 得到最合適的数字
		groupOutput.append(str(np.argmax(scores)))

	# 畫出来
	cv2.rectangle(imge, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(imge, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# 得到结果
	output.extend(groupOutput)

# 打印结果
print("Credit Card #: {}".format("".join(output)))
cv_show("Image", imge)
