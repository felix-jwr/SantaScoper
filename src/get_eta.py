import cv2


def get_eta(img_path):
    eta = "ETA: December 25th"
    img = cv2.imread(img_path)
    pos = (10, 30)

    text_size, _ = cv2.getTextSize(eta, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width, text_height = text_size
    padding = 5

    x, y = pos
    top_left = (x - padding, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + padding)

    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), -1)
    cv2.putText(img, eta, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(img_path, img)

