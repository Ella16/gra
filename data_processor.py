import os
from tqdm import tqdm 
import PyPDF2 # 표 모양은 인식 안되고, 글자만 인식됨 
import pdfplumber
raw_data_dir = 'data/raw/250314'
data_dir = 'data/processed/250314'
folder_names = {'regulations': '법령지침', 
                'guide_common': '안내서.지침-공통', 
                'guide_medicine': '안내서.지침-의약품',
                'guide_bio': '안내서.지침-바이오', 
                'guide_machine': '안내서.지침-의료기기',                 
                'guide_etc': '안내서.지침-기타', }
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for k, v in folder_names.items():
    if not os.path.exists(data_dir+'/'+v):
        os.makedirs(data_dir+'/'+v)

def parse_pdf_pypdf2(filepath):
    with open(filepath, "rb") as pdf:
        reader = PyPDF2.PdfReader(pdf)
        text:str = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def parse_pdf_inline(filepath):
    with pdfplumber.open(filepath) as pdf:
        text:str = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def parse_pdf_separate_tables(filepath):
    # table만 떼서 맨 마직에 따로 붙임 # 이게 효과가 있을까? # 이렇게 하려면 표를 json, csv 등으로 바꾸는 거 까지 해야 할듯 
    with pdfplumber.open(filepath) as pdf:
        table_text = []
        text = ""
        for page_id, page in enumerate(pdf.pages): 
            cur_text = page.extract_text() # 전체 페이지 파싱해서 

            table_bboxes = [table.bbox for table in page.find_tables()]
            for bbox in table_bboxes: # 표에 해당하는 텍스트를 제거
                cropped_page = page.within_bbox(bbox) 
                table_text = cropped_page.extract_text() 
                cur_text = cur_text.replace(table_text, "")
                table_text.append([page_id, table_text])
            text += cur_text
    text += '\n\n' + '\n'.join(table_text) 
    return text 

def pdf_to_txt():
    target_folders = ["guide_medicine"]
    for key in target_folders:
        src_dir = os.path.join(raw_data_dir, folder_names[key])
        dest_dir = os.path.join(data_dir, folder_names[key])
        pdf_files = [file for file in os.listdir(src_dir) if file.lower().endswith('.pdf')]
        processed_files = [file.replace('.txt', '.pdf') for file in os.listdir(dest_dir) if file.lower().endswith('.txt')]
        processed_files +=["27. 의료제품+임상통계+상담사례집.pdf",
                           "15. 항암제+비임상시험+가이드라인+질의응답집.pdf", #아.. 얘는 벌써 해버렸군 
                           "30. 2021+임상시험+관련+자주묻는+질의응답.pdf"
                           ] # 얘들은은 좋은 문서라서 따로 처리 
        pdf_files = list(set(pdf_files) - set(processed_files))

        # pdf_files =[ "1. 니자티딘+의약품+중+NDMA+분석법_공개.pdf",
        #              "10. 시판+후+의약품+위해성+관리계획+수립+시+약물유전체+활용+가이드라인.pdf",
        #             "11. 의약품+임상시험+대조군+설정+가이드라인.pdf",]
        N = len(pdf_files)
        for i, file in enumerate(pdf_files):
            text = parse_pdf_inline(os.path.join(src_dir, file))
            if len(text) > 0:
                with open(os.path.join(dest_dir, file.replace('.pdf', '.txt')), "w", encoding='utf-8') as txt:
                    txt.write(text)
            else:
                print(f'fail to parse: {src_dir}/{file}')            
            if i%10 ==0: 
                print(i/N)
        print(f'converting done: {src_dir}')
                
    print(text)

pdf_to_txt()