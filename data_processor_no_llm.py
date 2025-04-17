import os
# from tqdm import tqdm 
import PyPDF2 # 표 모양은 인식 안되고, 글자만 인식됨 
import pdfplumber

# def set_dir():
#     # raw_data_dir = 'data/raw/250314'
#     # data_dir = 'data/processed/250314'
#     # folder_names = {'regulations': '법령지침', 
#     #                 'guide_common': '안내서.지침-공통', 
#     #                 'guide_medicine': '안내서.지침-의약품',
#     #                 'guide_bio': '안내서.지침-바이오', 
#     #                 'guide_machine': '안내서.지침-의료기기',                 
#     #                 'guide_etc': '안내서.지침-기타', }
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     for k, v in folder_names.items():
#         if not os.path.exists(data_dir+'/'+v):
#             os.makedirs(data_dir+'/'+v)



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

def parse_pdf_inline(filepath):
    with pdfplumber.open(filepath) as pdf:
        text:str = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def parse_pdf_pypdf2(filepath):
    with open(filepath, "rb") as pdf:
        reader = PyPDF2.PdfReader(pdf)
        text:str = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def pdf_to_txt():
    # target_folders = ["guide_medicine"]
    data_dir = 'data-2504/qa/'
    target_folders = ["순환신경계약품과","약효동등성과", "의약품규격과","첨단의약품품질심사과"]
    
        
    import pandas as pd
    target = pd.read_csv(os.path.join(data_dir, 'target.csv'))
    target = target[~target['target'].isna()] # drop null 
    target.pdf = [f.replace(' ', '+') for f in target.pdf.values]
    
    
    for key in target_folders:
        src_dir = os.path.join(data_dir, 'pdf', key)
        dest_dir = os.path.join(data_dir, 'processed', key)
        os.makedirs(dest_dir, exist_ok=True)
        
        print('START to parse', src_dir )
        already_done = os.listdir(dest_dir)
        print('already done', already_done)
        pdf_files = [file for file in os.listdir(src_dir) if file.lower().endswith('.pdf') and file in target.pdf.values and file not in already_done]
        
        N = len(pdf_files)
        print('gonna parse', N, 'files')
        print('-----------', pdf_files)
        for i, file in enumerate(pdf_files):
            # text = parse_pdf_inline(os.path.join(src_dir, file))
            text = parse_pdf_pypdf2(os.path.join(src_dir, file))
            if len(text) > 0:
                with open(os.path.join(dest_dir, file.replace('.pdf', '.txt')), "w", encoding='utf-8') as txt:
                    txt.write(text)
                print(f'pass to parse: {src_dir}/{file}')            
            else:
                print(f'fail to parse: {src_dir}/{file}')            
            
        print(f'converting done: {src_dir}')
                
    # print(text)
    
def parse_qa_doc():
    data_dir = 'data-2504/qa/processed'
    target_folders = ["순환신경계약품과","약효동등성과", "의약품규격과","첨단의약품품질심사과"]
    qadocs = []
    for f in target_folders:
        sub_f = os.path.join(data_dir, f)
        qadocs += [ os.path.join(sub_f, doc) for doc in os.listdir(sub_f) if doc.endswith('txt')]
        
    import pandas as pd
    target = pd.read_csv(os.path.join('data-2504/qa/', 'target.csv'))
    # target = target[~target['target'].isna()] # drop null 
    # target = target[target.target != 'all-table']
    # target = target[target.target != 'all-comment']
    
    target = target[target.target == 'all']
    target['txt'] = [f.replace(' ', '+').replace('.pdf', '.txt') for f in target.pdf.values]
      
    
    res = []
    for i, r in target.iterrows():
        doc_path = os.path.join(data_dir, r['part'], r['txt'])
        if not os.path.exists(doc_path): continue

        print('start', doc_path)
    
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc = f.readlines()
            doc = '{newline}'.join(doc)
        
        doc_name = r['txt']
        queries = r['target'].split(',')
        queries = [q.strip().lower() for q in queries]    
        

        # q = [] # 한 질문에 
        # a = [] # 답이 여러 체크포인트로 이뤄짐 
        
        def _find_start(qv, num):
            start = -1
            qnum = qv%(num)
            i = doc.find(qnum) 
            if i > -1:
                while doc[i-9:i] != '{newline}' and i != -1:
                    i = doc.find(qnum, i+1) 
                start = i
            
            return start
        
        ALL = True
        qa_list = []
        if ALL:
            
            num =  1
            start = -1
            cur_qv = ''
            qvars = ['Q%s', 'Q %s', '질문 %s', '질문%s']
            doc_len = len(doc)
            for qv in qvars:
                start = _find_start(qv, num)
                if start != -1:
                    cur_qv = qv
                    break
                
            while start != -1 and start < doc_len:
                
                end = _find_start(cur_qv,str(num+1))
                qa = doc[start:end]
                qa_list.append([doc_name, qv%num, qa])
                
                start = end
                num += 1
            res += qa_list
        else:
            
            print('#queries', len(queries), doc_name, queries)
            doc_len = len(doc)
            for q in queries:
                num = q[1:]
                qvars = ['Q%s', 'Q %s', '질문 %s', '질문%s']
                
                
                def _find_start(qv, num):
                    start = -1
                    qnum = qv%(num)
                    i = doc.find(qnum) 
                    if i > -1:
                        while doc[i-9:i] != '{newline}' and i != -1:
                            i = doc.find(qnum, i+1) 
                        start = i
                    
                    return start
                    
                for qv in qvars:
                    if '-' not in num:
                        num_1 = str(int(num) + 1)
                    else:
                        j = num.find('-')
                        num_1 = num[:j+1] + str(int(num[j+1:]) + 1)
                        
                    start = _find_start(qv, num)    
                    if start > -1:    
                        end = _find_start(qv, num_1)
                        qa = doc[start:end]
                        qa_list.append([doc_name, q, qa])
                        break
                
            res+=qa_list
                            
            print('#saved', len(qa_list), '/', len(queries))                       
            
            
        y = pd.DataFrame(res, columns=['doc_name', 'query', 'qa'])    
        y.to_csv('qa-doc-targetall.csv', index=False)            
    
        
#queries 3 20231127_국소+적용+용해성+마이크로니들+일반의약품+개발+질의·응답집_민원인안내서_최종.txt
#queries 15 붙임1_「복합제+임상시험+가이드라인」+(민원인안내서)+개정+(1).txt
#queries 1 안내서-0187-02+「고령자+대상+임상시험+가이드라인」.txt
#queries 1 안내서-0194-02+「의약품+국제공통기술문서(CTD)+해설서」[민원인+안내서].txt
#queries 15 알츠하이머형+치매+치료+복합제+개발+관련+질의·응답집.txt
#queries 9 BCS에+따른+생동성시험+면제를+위한+제출자료+작성방법+가이드라인+QnA+개정.txt
#queries 13 산제+및+과립제의+의약품동등성시험에+대한+질의응답집(민원인+안내서+제정).txt
#queries 91 의약품동등성시험+이백문이백답+자주묻는+질의응답집(민원인안내서)_201803.txt
#queries 1 180508_원료의약품의_개발_및_제조_품질심사_가이드라인_질의응답집(안)_배포용.txt
#queries 1 국제공통기술문서+작성+질의응답집(ICH+M4+QNA).txt
#queries 1 의약품+허가+후+제조방법+변경관리+질의응답집(민원인안내서).txt
#queries 12 의약품등+안정성시험기준+질의응답집[민원인+안내서].txt
#queries 1 제네릭의약품_국제공통기술문서(CTD)_질의응답집_개정_2021.5._최종.txt
#queries 1 의약품+국제공통기술문서+작성을+위한+질의응답집(품질)(민원인안내서).txt

if __name__=="__main__":
    # pdf_to_txt()
    parse_qa_doc()



# processed_files = [file.replace('.txt', '.pdf') for file in os.listdir(dest_dir) if file.lower().endswith('.txt')]
        # processed_files +=["27. 의료제품+임상통계+상담사례집.pdf",
        #                    "30. 2021+임상시험+관련+자주묻는+질의응답.pdf"
        #                 #    "15. 항암제+비임상시험+가이드라인+질의응답집.pdf", 
        #                    # "53-2. 의약품+허가+후+제조방법+변경관리+질의응답집(민원인안내서).pdf",
        #                    ] # 얘들은은 좋은 문서라서 따로 처리 
        # pdf_files = list(set(pdf_files) - set(processed_files))
        # pdf_files = ["27. 의료제품+임상통계+상담사례집.pdf",
        #                    "30. 2021+임상시험+관련+자주묻는+질의응답.pdf"]

        # pdf_files =[ "1. 니자티딘+의약품+중+NDMA+분석법_공개.pdf",
        #              "10. 시판+후+의약품+위해성+관리계획+수립+시+약물유전체+활용+가이드라인.pdf",
        #             "11. 의약품+임상시험+대조군+설정+가이드라인.pdf",]