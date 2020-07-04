import cv2
import numpy as np
import pytesseract
import asyncio

async def recognize_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(gray, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img_erosion, config=custom_config)
    print(text)


def main():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        asyncio.run(recognize_text(frame))

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()