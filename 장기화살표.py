import cv2
import numpy as np
import math
import mss

def capture_screenshot(path = 'tmp_screenshot.png'):
    with mss.mss() as sct:
        img = sct.shot(output = path)

# 이미지 및 라인 탐지 코드
capture_screenshot('tmp_screenshot.png')
image = cv2.imread('tmp_screenshot.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
origin_image = image.copy()



def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if denom == 0:
                continue
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            if 0 <= px <= image.shape[1] and 0 <= py <= image.shape[0]:
                intersections.append((int(px), int(py)))
    return intersections

intersections = find_intersections(lines)

# 마우스 이벤트 처리
drawing = False
line_start = None
current_line = None
lines_drawn = []
distance_threshold = 30  # 임의의 값. 격자점과의 거리의 1/3 수준을 원했음.
color = (0, 255, 0)  # 기본 색상(현재는 초록.)
ns, ne = (0,0), (0,0)

def closest_intersection(point, intersections, threshold):
    for (x, y) in intersections:
        if np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) <= threshold:
            return (x, y)
    return None

def draw_arrow(image, start, end, color):
    distance = round(math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2),2)

    # 거리로 tipLength 계산
    tip_l = round(30/distance, 2) if distance != 0 else 0.01 #30은 임의의 값.
    cv2.arrowedLine(image, start, end, color, 2, tipLength=tip_l)

#def draw_arrow2(image, start, end, color, t=0.9):
#    x1, y1 = start
#    x2, y2 = end
#    x = int(x1 + t * (x2 - x1))
#    y = int(y1 + t * (y2 - y1))
#    cv2.arrowedLine(image, (x1, y1), (x, y), color, 2, tipLength=0.05)
    
def draw_line(event, x, y, flags, param):
    global drawing, line_start, intersections, lines_drawn, image, current_line, color, ns, ne
    
    if event == cv2.EVENT_RBUTTONDOWN:
        
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            color = (255, 0, 0)  # Ctrl 키가 눌린 경우 색상 변경
        else:
            color = (0, 255, 0)  # 기본 색상
            
        nearest_start = closest_intersection((x, y), intersections, distance_threshold)
        ns = nearest_start
        if nearest_start:
            line_start = nearest_start
            drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = image.copy()
            nearest_end = closest_intersection((x, y), intersections, distance_threshold)
            if nearest_end:
                current_line = (line_start, nearest_end)
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    cv2.line(temp_image, line_start, nearest_end, color, 2)
                else:
                    draw_arrow(temp_image, line_start, nearest_end, color)
            cv2.imshow('Image', temp_image)

    elif event == cv2.EVENT_RBUTTONUP:
        if drawing:
            drawing = False
            if current_line:
                lines_drawn.append(current_line)
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    cv2.line(image, current_line[0], current_line[1], color, 2)
                else:
                    draw_arrow(image, current_line[0], current_line[1], color)
                current_line = None
            nearest_end = closest_intersection((x, y), intersections, distance_threshold)
            ne = nearest_end
            if ns == ne:
                # 격자점 주변에 원 또는 X자 그리기
                numr = 25#도형 크기 지정
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    cv2.line(image, (ns[0]-numr, ns[1]-numr), (ns[0]+numr, ns[1]+numr), color, 2)
                    cv2.line(image, (ns[0]+numr, ns[1]-numr), (ns[0]-numr, ns[1]+numr), color, 2)
                else:
                    cv2.circle(image, (ns[0], ns[1]), numr, color, 2)
            cv2.imshow('Image', image)

    elif event == cv2.EVENT_LBUTTONDOWN:
        image[:] = origin_image[:]  # 원래 이미지로 복원
        lines_drawn.clear()
        cv2.imshow('Image', image)

# 격자점 표시
for (x, y) in intersections:
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

temp_image = image.copy()
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', draw_line)

while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC 키를 눌러야만 종료
        break

cv2.destroyAllWindows()
