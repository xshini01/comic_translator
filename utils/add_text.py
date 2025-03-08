from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2

def add_text(image, text, font_path, bubble_contour):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    line_height = 16
    font_size = 14
    wrapping_ratio = 0.075

    max_iterations = 100
    iterations = 0

    while iterations < max_iterations:
        wrapped_text = textwrap.fill(text, width=max(3, int(w * wrapping_ratio)), break_long_words=False)
        font = ImageFont.truetype(font_path, size=font_size)
        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height

        if total_text_height <= h and all(draw.textlength(line, font=font) <= w for line in lines):
            if total_text_height < h * 0.8 and font_size < 48 and all(draw.textlength(line, font=font) <= 0.9 * w for line in lines):
                wrapping_ratio = max(0.02, wrapping_ratio - 0.005)
                line_height += 2
                font_size += 2

                if total_text_height == line_height and all(draw.textlength(line, font=font) >= 0.5 * w for line in lines):
                    break

            else:
                break  

        elif total_text_height > h or any(draw.textlength(line, font=font) > w for line in lines):
            line_height = max(16, line_height -2)
            font_size = max(14, font_size - 2)
            wrapping_ratio = min(wrapping_ratio + 0.01, 0.1)

        iterations += 1

    text_y = y + (h - total_text_height) // 2

    for line in lines:
        text_length = draw.textlength(line, font=font)
        text_x = x + (w - text_length) // 2
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
