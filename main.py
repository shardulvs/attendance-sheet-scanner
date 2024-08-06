import cv2
import numpy as np
import subprocess
import output
process_count = 0


def store_process_image(file_name, image):
    global process_count
    path = f"./processing_steps/{process_count}_{file_name}.jpg"
    cv2.imwrite(path, image)
    process_count += 1


def calculate_distance_between_2_points(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


path_to_image = "./image/att.jpg"
image = cv2.imread(path_to_image)
store_process_image("original_image", image)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
store_process_image("grayscale_image", grayscale_image)
threshold_image = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)[1]
store_process_image("threshold_image", threshold_image)
invert_image = cv2.bitwise_not(threshold_image)
store_process_image("invert_image", invert_image)
dilate_image = cv2.dilate(invert_image, np.ones((3, 3), np.uint8), iterations=5)
store_process_image("dilate_image", dilate_image)
contours, hierarchy = cv2.findContours(dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dilate_bgr = cv2.cvtColor(dilate_image, cv2.COLOR_GRAY2BGR)
image_with_all_contours = cv2.drawContours(dilate_bgr.copy(), contours, -1, (0, 255, 0), 3)
store_process_image("image_with_all_contours", image_with_all_contours)
rectangular_contours = []
the_count = 0
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        rectangular_contours.append(approx)
image_with_only_rectangular_contours = cv2.drawContours(dilate_bgr.copy(), rectangular_contours, -1, (0, 255, 0), 3)
store_process_image("image_with_only_rectangular_contours", image_with_only_rectangular_contours)
max_area = 0
contour_with_max_area = None
for contour in rectangular_contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        contour_with_max_area = contour
image_with_contour_with_max_area = cv2.drawContours(dilate_bgr.copy(), [contour_with_max_area], -1, (0, 255, 0), 3)
store_process_image("image_with_contour_with_max_area", image_with_contour_with_max_area)
corner_points = contour_with_max_area.reshape(4, 2)
# clockwise starting from top-left
corner_points_ordered = np.zeros((4, 2), dtype="float32")
# origin is at top left
# top-left --> the smallest sum (x + y)
# bottom-right --> largest sum (x + y)
s = corner_points.sum(axis=1)
corner_points_ordered[0] = corner_points[np.argmin(s)]
corner_points_ordered[2] = corner_points[np.argmax(s)]
# top-right --> smallest difference (y - x)
# bottom-left --> largest difference (y - x)
diff = np.diff(corner_points, axis=1)
corner_points_ordered[1] = corner_points[np.argmin(diff)]
corner_points_ordered[3] = corner_points[np.argmax(diff)]
image_with_corner_points = cv2.cvtColor(threshold_image.copy(), cv2.COLOR_GRAY2BGR)
for point in corner_points_ordered:
    point_coordinates = (int(point[0]), int(point[1]))
    cv2.circle(image_with_corner_points, point_coordinates, 30, (0, 0, 255), 5)
    cv2.circle(image_with_corner_points, point_coordinates, 10, (0, 0, 255), -1)
store_process_image("image_with_corner_points", image_with_corner_points)
original_image_width = image.shape[1]
table_width = calculate_distance_between_2_points(corner_points_ordered[0], corner_points_ordered[1])
table_height = calculate_distance_between_2_points(corner_points_ordered[0], corner_points_ordered[3])
new_image_width = int(original_image_width * 0.9)
new_image_height = int(new_image_width * table_height / table_width)
pts_1 = np.float32(corner_points_ordered)
pts_2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
perspective_corrected_image = cv2.warpPerspective(threshold_image, matrix, (new_image_width, new_image_height))
store_process_image("perspective_corrected_image", perspective_corrected_image)
padding = int(new_image_height * 0.1)
perspective_corrected_image_with_padding = cv2.copyMakeBorder(perspective_corrected_image, padding, padding, padding,
                                                              padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
corrected_image_invert = cv2.bitwise_not(perspective_corrected_image_with_padding)
store_process_image("perspective_corrected_image_with_padding", perspective_corrected_image_with_padding)
lines = cv2.HoughLinesP(corrected_image_invert, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=10)
corrected_image_invert_lines_detected = cv2.cvtColor(corrected_image_invert, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(corrected_image_invert_lines_detected, (x1, y1), (x2, y2), (0, 255, 0), 30)
        cv2.line(corrected_image_invert, (x1, y1), (x2, y2), (0, 0, 0), 30)
store_process_image("corrected_image_invert_lines_detected", corrected_image_invert_lines_detected)
corrected_image_invert_lines_removed = corrected_image_invert
store_process_image("corrected_image_invert_lines_removed", corrected_image_invert_lines_removed)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
corrected_image_invert_lines_noise_removed = cv2.erode(corrected_image_invert_lines_removed, kernel, iterations=1)
corrected_image_invert_lines_noise_removed = cv2.dilate(corrected_image_invert_lines_noise_removed, kernel,
                                                        iterations=1)
corrected_image_lines_noise_removed = cv2.bitwise_not(corrected_image_invert_lines_noise_removed)
store_process_image("corrected_image_invert_lines_noise_removed", corrected_image_invert_lines_noise_removed)
store_process_image("corrected_image_lines_noise_removed", corrected_image_lines_noise_removed)
kernel_to_remove_gaps_between_words = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
corrected_dilate_image = cv2.dilate(corrected_image_invert_lines_noise_removed, kernel_to_remove_gaps_between_words,
                                    iterations=10)
store_process_image("corrected_dilate_image", corrected_dilate_image)
text_contours, _ = cv2.findContours(corrected_dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_text_contours = cv2.cvtColor(corrected_image_invert_lines_noise_removed, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_text_contours, text_contours, -1, (0, 255, 0), 3)
store_process_image("image_with_text_contours", image_with_text_contours)
bounding_boxes = []
image_with_all_bounding_boxes = cv2.cvtColor(corrected_image_invert_lines_noise_removed, cv2.COLOR_GRAY2BGR)
for contour in text_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w*h > 10000:
        bounding_boxes.append((x, y, w, h))
        image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
store_process_image("image_with_all_bounding_boxes", image_with_all_bounding_boxes)
heights_of_bounding_boxes = []
for bounding_box in bounding_boxes:
    x, y, w, h = bounding_box
    heights_of_bounding_boxes.append(h)
mean_height = np.mean(heights_of_bounding_boxes)
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
rows = []
half_of_mean_height = mean_height / 2
current_row = [bounding_boxes[0]]
for bounding_box in bounding_boxes[1:]:
    current_bounding_box_y = bounding_box[1]
    previous_bounding_box_y = current_row[-1][1]
    distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
    if distance_between_bounding_boxes <= half_of_mean_height:
        current_row.append(bounding_box)
    else:
        rows.append(current_row)
        current_row = [bounding_box]
rows.append(current_row)
for row in rows:
    row.sort(key=lambda x: x[0])


def get_result_from_tesseract(image_path):
    tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'
    output = subprocess.getoutput(
        f'{tesseract_cmd} {image_path}' + ' - -l eng --oem 3 --psm 7 --dpi 72 -c '
                                          'tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "')
    output = output.strip()
    return output


table = []
current_row = []
image_number = 0
for row in rows:
    for bounding_box in row:
        x, y, w, h = bounding_box
        y = y - 5
        cropped_image = corrected_image_lines_noise_removed[y:y + h, x:x + w]
        image_slice_path = "./text_images_for_ocr/img_" + str(image_number) + ".jpg"
        cv2.imwrite(image_slice_path, cropped_image)
        results_from_ocr = get_result_from_tesseract(image_slice_path)
        current_row.append(results_from_ocr)
        image_number += 1
    table.append(current_row)
    current_row = []
output.print_2d_array(table)
with open("final_output.csv", "w") as f:
    for row in table:
        f.write(",".join(row) + "\n")
