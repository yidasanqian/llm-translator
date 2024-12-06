import os
import pdfplumber
from openai import AzureOpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from logging import getLogger
import re
import tempfile

# 配置日志级别
logger = getLogger(__name__)
logger.setLevel('INFO')

# 设置 OpenAI API 密钥
deployment_name='gpt-4o' # 部署的模型名称


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def chat_completions(messages, model_name: str=deployment_name, tempreture=0.0):   
    response = client.chat.completions.create(model=model_name, 
                                              messages=messages, 
                                              max_tokens=4096,
                                              temperature=tempreture, 
                                              )
    result = response.choices[0].message.content   
    return result

# 字体配置，包括字体、大小比例和类型
LANGUAGE_CONFIG = {
    'en': {'font': '微软雅黑', 'scale': 1.0, 'font_style': 'normal'},
    'zh-cn': {'font': '微软雅黑', 'scale': 0.9, 'font_style': 'normal'},
}

# 注册字体
font_filename = "fonts/微软雅黑.ttf"
current_path = os.path.dirname(os.path.abspath(__file__))
pdfmetrics.registerFont(TTFont("微软雅黑", os.path.join(current_path, font_filename)))

# 使用 OpenAI GPT 模型进行翻译
def translate_text(text, source_lang='en', target_lang='zh-cn'):
    prompt=f"""
    Translate the following text from {source_lang} to {target_lang}: {text}
    """
    messages = [   
                {"role": "system", "content": "你是一个世界级的多语言翻译系统。无需翻译数学表达式、代码、数字"},                
        {"role": "user", "content": prompt}
    ]
    try:
        return chat_completions(messages, tempreture=0.7)
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        return text  # 失败时返回原文

# 检查文本是否为中文
def is_chinese(text):
    return re.search('[\u4e00-\u9fff]', text) is not None

# 获取语言配置
def get_language_config(text):
    return LANGUAGE_CONFIG['zh-cn'] if is_chinese(text) else LANGUAGE_CONFIG['en']

# 自适应字体大小
def get_adaptive_font_size(c, text, max_width, font_name, initial_size):
    font_size = initial_size
    while c.stringWidth(text, font_name, font_size) > max_width:
        font_size -= 0.5
    return font_size

# 动态列检测
def detect_dynamic_columns(words):
    columns = {}
    for word in words:
        x_center = (word['x0'] + word['x1']) / 2
        matched = False
        for key, (x0, x1) in columns.items():
            if abs(x_center - (x0 + x1) / 2) < 20:  # 允许偏差
                columns[key] = (min(x0, word['x0']), max(x1, word['x1']))
                matched = True
                break
        if not matched:
            columns[x_center] = (word['x0'], word['x1'])
    sorted_columns = sorted(columns.values(), key=lambda x: x[0])
    return sorted_columns

# 智能行间距调整
def calculate_line_spacing(text_height, density_factor=1.0):
    """根据文本高度和内容密度调整行间距"""
    base_spacing = 1.2 if density_factor < 0.5 else 1.5  # 密集内容增加行间距
    return text_height * base_spacing * density_factor

# 内容密度检测
def detect_content_density(words, page_height):
    """根据页面内容密集度返回一个密度系数（0.5-1.5范围内），用于行间距调整"""
    content_height = sum([word['bottom'] - word['top'] for word in words])
    density_factor = min(1.5, max(0.5, content_height / page_height))  # 限制密度系数范围
    return density_factor



def handle_pdf(trans_page, input_pdf_path, output_pdf_path, source_lang, target_lang):
    # 创建PDF画布
    c = canvas.Canvas(output_pdf_path, pagesize=letter)

    # 处理PDF
    with pdfplumber.open(input_pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            # 判断是否需要翻译,trans_page为要翻译的页码，例如：1表示翻译第一页，1-3表示翻译第1到第3页
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
            
            page = pdf.pages[page_num]
            page_width, page_height = page.width, page.height
            c.setPageSize((page_width, page_height))
            
            words = page.extract_words(keep_blank_chars=True)            

            for text_box in words:
                text = text_box['text']
                if text.strip() == '':
                    continue
                
                x0, y0, x1 = text_box['x0'], text_box['top'], text_box['x1']
                font_size = text_box['height']
                translated_text = translate_text(text, source_lang, target_lang)
                config = get_language_config(translated_text)
                font_name = config['font']
                adjusted_font_size = get_adaptive_font_size(c, translated_text, x1 - x0, font_name, font_size * config['scale'])
                c.setFont(font_name, adjusted_font_size)
                # 在 PDF 中，y0 表示顶部的距离，需要转换为 reportlab 的坐标，即 adjusted_y0 = page_height - y0。
                adjusted_y0 = page_height - y0
                # 绘制文本
                c.drawString(x0, adjusted_y0, translated_text)

            # 处理图像
            for image in page.images:
                img_x0, img_y0 = image['x0'], image['top']
                img_width, img_height = image['width'], image['height']
                adjusted_img_y0 = page_height - img_y0 - img_height
                img_bytes = BytesIO()
                img = page.within_bbox((img_x0, img_y0, img_x0 + img_width, img_y0 + img_height)).to_image()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # 使用临时文件保存图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_bytes.getvalue())
                    tmp_file_path = tmp_file.name
                
                c.drawImage(tmp_file_path, img_x0, adjusted_img_y0, img_width, img_height)
                os.remove(tmp_file_path)  # 删除临时文件
            # 下一页
            c.showPage()           

    # 保存PDF
    c.save()


if __name__ == '__main__': 
    # PDF文件路径
    input_pdf_path = './doc/2410.13085v1.pdf'
    output_pdf_path = './output/2410.13085v1.pdf'
    source_lang = 'en'
    target_lang = 'zh-cn'

    # 要翻译的页码，例如：1表示翻译第一页，1-3表示翻译第1到第3页
    trans_page = "1"
    output_pdf_path = output_pdf_path.replace('.pdf', f'_{source_lang}_{target_lang}({trans_page}).pdf')
    handle_pdf(trans_page, input_pdf_path, output_pdf_path, source_lang, target_lang)

    logger.info("PDF翻译一页完成！")
