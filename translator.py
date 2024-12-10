import os
import re
import pdfplumber
from openai import AzureOpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from logging import getLogger
import tempfile
import math

# 配置日志
logger = getLogger(__name__)
logger.setLevel('INFO')

# 设置 OpenAI API
deployment_name = "gpt-4o"
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# 字体配置
LANGUAGE_CONFIG = {
    'en': {'font': '微软雅黑', 'scale': 1.0, 'font_style': 'normal'},
    'zh-cn': {'font': '微软雅黑', 'scale': 0.9, 'font_style': 'normal'},
}
font_filename = "fonts/微软雅黑.ttf"
current_path = os.path.dirname(os.path.abspath(__file__))
pdfmetrics.registerFont(TTFont("微软雅黑", os.path.join(current_path, font_filename)))

# 使用 LLM 翻译
def translate_text(text, source_lang='en', target_lang='zh-cn'):
    prompt = f"Translate the following text from {source_lang} to {target_lang}: {text}"
    messages = [
        {"role": "system", "content": "你是一个世界级的多语言翻译系统。当文本包含数学表达式、数字、代码或标识符直接返回原始内容。"},
        {"role": "user", "content": prompt}
    ]
    try:
        return chat_completions(messages, tempreture=0.7)
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        return text

def chat_completions(messages, model_name=deployment_name, tempreture=0.7):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        temperature=tempreture,
    )
    return response.choices[0].message.content.strip()

# 提取数学表达式的正则表达式
MATH_PATTERN = re.compile(r'(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\))')

def process_text_with_math(text):
    """
    检测文本中的数学表达式，避免翻译 LaTeX 数学公式。
    返回: 一个包含文本和数学公式的混合列表。
    """
    segments = []
    last_end = 0
    for match in MATH_PATTERN.finditer(text):
        start, end = match.span()
        # 添加前面普通文本
        if start > last_end:
            segments.append(("text", text[last_end:start]))
        # 添加数学公式
        segments.append(("math", match.group()))
        last_end = end
    # 添加剩余普通文本
    if last_end < len(text):
        segments.append(("text", text[last_end:]))
    return segments

# 检查文本是否包含中文字符
def is_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)


# 自适应字体大小
def get_adaptive_font_size(c, text, max_width, font_name, initial_size):
    font_size = initial_size
    while c.stringWidth(text, font_name, font_size) > max_width:
        font_size -= 0.5
    return font_size

def is_vertical_text(chars):
    """
    检测字符是否为竖排文字，根据 matrix 矩阵判断。
    char对象：
    {'matrix': (0, 1, -1, 0, 32.0, 568.62), 'fontname': 'Times-Roman', 'adv': 10.0, 'upright': False, 'x0': 16.34, 'y0': 568.62, 'x1': 36.34, 'y1': 578.62, 'width': 20.000000000000004, 'height': 10.0, 'size': 10.0, 'mcid': None, 'tag': None, 'object_type': 'char', 'page_number': 1, 'ncs': 'DeviceGray', 'text': '4', 'stroking_color': (0.5,), 'stroking_pattern': None, 'non_stroking_color': (0.5,), 'non_stroking_pattern': None, 'top': 213.38, 'bottom': 223.38, 'doctop': 213.38}
    """
    if not chars:
        return False
    for char in chars:
        matrix = char.get('matrix', [])
        if len(matrix) >= 6 and matrix[0] == 0 and matrix[3] == 0 and (matrix[1] != 0 or matrix[2] != 0):
            return True
    return False


def draw_text(c, text, x, y, font_name, chars):
    """
    绘制文字，可处理竖排方向。
    """  
    is_vertical = is_vertical_text(chars)
    if is_vertical:
        y_offset = y
        for char in chars:
            matrix = char.get('matrix', [])
            if not matrix or len(matrix) < 6:
                print(f"Invalid matrix for character: {char}")
                continue
            
            # 获取旋转角度
            a, b, _c, d, e, f = matrix
            angle = math.degrees(math.atan2(b, a))
            
            # 设置字体和大小
            c.setFont(font_name, char['size'])
            
            # 保存状态，旋转并绘制字符
            c.saveState()
            c.translate(x, y_offset)  # 移动到字符的起始位置
            c.rotate(angle)  # 旋转字符
            c.drawString(0, 0, char['text'])  # 绘制字符
            c.restoreState()
            
            # 更新y_offset，用于下一个字符的位置
            y_offset -= char['size']  # 根据字符大小调整间距  
    else:
        c.drawString(x, y, text)


def handle_pdf(trans_page, input_pdf_path, output_pdf_path, source_lang, target_lang):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)

    with pdfplumber.open(input_pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 检查是否需要翻译当前页
            if trans_page:
                trans_page_range = trans_page.split('-')
                if len(trans_page_range) == 1:
                    start_page = end_page = int(trans_page_range[0]) - 1
                elif len(trans_page_range) == 2:
                    start_page = int(trans_page_range[0]) - 1
                    end_page = int(trans_page_range[1]) - 1
                else:
                    raise ValueError("Invalid page range format. Use '1' or '1-3'.")

                if not (start_page <= page_num <= end_page):
                    continue

            # 设置页面大小
            page_width, page_height = page.width, page.height
            c.setPageSize((page_width, page_height))

            # 提取文本行（保留空行和空格）
            lines = page.extract_text_lines(keep_blank_chars=True)
            # 取后5行
            lines = lines[-1:]
            for line in lines:
                text = line['text']
                chars = line.get('chars', [])
                if text.strip() == '':
                    translated_text = text  # 空行保持原样
                else:
                    segments = process_text_with_math(text)
                    translated_text = ""
                    for segment_type, content in segments:
                        if segment_type == "math":
                            translated_text += content  # 数学公式保持原样
                        else:
                            translated_text += translate_text(content, source_lang, target_lang)

                # 检查文本方向并翻转 Y 坐标
                x0, y0 = line['x0'], page_height - line['top']
                font_size = line['bottom'] - line['top']
                config = LANGUAGE_CONFIG['zh-cn'] if is_chinese(translated_text) else LANGUAGE_CONFIG['en']
                font_name = config['font']
                adjusted_font_size = get_adaptive_font_size(c, translated_text, line['x1'] - line['x0'], font_name, font_size * config['scale'])

                c.setFont(font_name, adjusted_font_size)

                # 绘制文本        
                draw_text(c, translated_text, x0, y0, font_name, chars)

                # 写入进度
                with open("output/translation_progress.txt", "a", encoding="utf-8") as progress_file:
                    progress_file.write(f"Page {page_num + 1}, Line at x={line['x0']}, y={line['top']}: {text} -> {translated_text}\n")

            # 处理图像
            for image in page.images:
                img_x0, img_y0 = image['x0'], image['top']
                img_width, img_height = image['width'], image['height']
                adjusted_img_y0 = page_height - img_y0 - img_height
                img_bytes = BytesIO()
                img = page.within_bbox((img_x0, img_y0, img_x0 + img_width, img_y0 + img_height)).to_image()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_bytes.getvalue())
                    tmp_file_path = tmp_file.name

                c.drawImage(tmp_file_path, img_x0, adjusted_img_y0, img_width, img_height)
                os.remove(tmp_file_path)

            # 下一页
            c.showPage()

    c.save()
        
# 主程序
if __name__ == '__main__':
    input_pdf_path = './doc/2410.13085v1.pdf'
    output_pdf_path = './output/2410.13085v1_translated.pdf'
    source_lang = 'en'
    target_lang = 'zh-cn'
    trans_page = "1"  # 设置翻译页码范围

    handle_pdf(trans_page, input_pdf_path, output_pdf_path, source_lang, target_lang)
    logger.info("PDF 翻译完成！")
