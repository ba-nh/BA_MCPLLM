import pdfplumber
import json
import re
from collections import defaultdict

def extract_tables_from_pdf(pdf_path: str):
    """PDF에서 표를 추출합니다."""
    all_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        for page_num in range(total_pages):
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if table and len(table) > 0:
                    cleaned_table = []
                    for row in table:
                        if row and any(cell and str(cell).strip() for cell in row):
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                    
                    if cleaned_table:
                        table_info = {
                            'page': page_num + 1,
                            'table_index': table_idx,
                            'rows': len(cleaned_table),
                            'columns': len(cleaned_table[0]) if cleaned_table else 0,
                            'data': cleaned_table
                        }
                        all_tables.append(table_info)
    
    return all_tables

def is_two_column_page(page) -> bool:
    """페이지가 2단 구조인지 판별"""
    words = page.extract_words(extra_attrs=["size", "fontname", "non_stroking_color", "stroking_color"])
    if not words:
        return False
    
    x_coords = [w['x0'] for w in words]
    center_x = (min(x_coords) + max(x_coords)) / 2
    
    # 1. 표가 중앙에 걸쳐있는지 확인 (우선)
    tables = page.extract_tables()
    if tables:
        for table in tables:
            if table_spans_center(page, center_x):
                return False  # 1단 구조
    
    # 2. 중앙에 단어가 있는지 확인
    for word in words:
        x, y = word['x0'], word['top']
        if y > 30 and abs(x - center_x) <= 4:
            return False  # 1단 구조
    
    return True  # 2단 구조

def table_spans_center(page, center_x, tolerance=10):
    """표가 중앙에 걸쳐있는지 확인"""
    tables = page.extract_tables()
    if not tables:
        return False
    
    # 각 표의 실제 위치를 확인
    for table in tables:
        if not table:
            continue
        
        # 표의 내용을 기반으로 위치 추정
        table_words = []
        for row in table:
            if row:
                for cell in row:
                    if cell and isinstance(cell, str):
                        # 표 셀 내용에서 단어 추출
                        words_in_cell = cell.split()
                        for word in words_in_cell:
                            if word.strip():
                                table_words.append(word.strip())
        
        if not table_words:
            continue
        
        # 표 주변의 단어들을 찾아서 표의 x 좌표 범위 추정
        page_words = page.extract_words(extra_attrs=["size", "fontname", "non_stroking_color", "stroking_color"])
        
        # 표 내용과 일치하는 단어들의 x 좌표 수집
        matching_x_coords = []
        for page_word in page_words:
            if page_word['text'] in table_words:
                matching_x_coords.append(page_word['x0'])
        
        if matching_x_coords:
            table_min_x = min(matching_x_coords)
            table_max_x = max(matching_x_coords)
            
            # 중앙이 표 영역에 포함되는지 확인
            if table_min_x - tolerance <= center_x <= table_max_x + tolerance:
                return True
    
    return False

def split_words_by_column(words, center_x):
    """단어들을 왼쪽/오른쪽 컬럼으로 분리"""
    left_words = []
    right_words = []
    
    for word in words:
        if word['x0'] < center_x:
            left_words.append(word)
        else:
            right_words.append(word)
    
    return left_words, right_words

def extract_font_sentences(pdf_path: str):
    """PDF에서 폰트 크기별로 문장을 추출합니다."""
    font_sentences = defaultdict(list)
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        for page_num in range(total_pages):
            page = pdf.pages[page_num]
            words = page.extract_words(extra_attrs=["size", "fontname", "non_stroking_color", "stroking_color"])
            
            # 2단/1단 구조 판별
            is_two_col = is_two_column_page(page)
            
            if is_two_col:
                # 2단 구조: 왼쪽/오른쪽 컬럼으로 분리하여 처리
                x_coords = [w['x0'] for w in words]
                center_x = (min(x_coords) + max(x_coords)) / 2
                left_words, right_words = split_words_by_column(words, center_x)
                
                # 왼쪽 컬럼 먼저 처리
                for column_words, column_name in [(left_words, "left"), (right_words, "right")]:
                    font_words = defaultdict(list)
                    for word in column_words:
                        font_size = word.get('size', 0)
                        font_words[font_size].append({
                            'text': word['text'],
                            'top': word.get('top', 0),
                            'x0': word.get('x0', 0),
                            'page': page_num + 1,
                            'column': column_name
                        })
                    
                    for font_size, word_list in font_words.items():
                        lines = defaultdict(list)
                        for word in word_list:
                            y0 = round(word['top'], 1)
                            lines[y0].append(word)
                        
                        for y0 in sorted(lines.keys()):
                            line_words = sorted(lines[y0], key=lambda x: x['x0'])
                            sentence = ' '.join([w['text'] for w in line_words])
                            
                            sentence = re.sub(r'\s+', ' ', sentence).strip()
                            if len(sentence) > 3:
                                font_sentences[font_size].append({
                                    'sentence': sentence,
                                    'page': line_words[0]['page'],
                                    'y0': y0,
                                    'column': column_name
                                })
            else:
                # 1단 구조: 기존 방식 그대로 사용
                font_words = defaultdict(list)
                for word in words:
                    font_size = word.get('size', 0)
                    font_words[font_size].append({
                        'text': word['text'],
                        'top': word.get('top', 0),
                        'x0': word.get('x0', 0),
                        'page': page_num + 1
                    })
                
                for font_size, word_list in font_words.items():
                    lines = defaultdict(list)
                    for word in word_list:
                        y0 = round(word['top'], 1)
                        lines[y0].append(word)
                    
                    for y0 in sorted(lines.keys()):
                        line_words = sorted(lines[y0], key=lambda x: x['x0'])
                        sentence = ' '.join([w['text'] for w in line_words])
                        
                        sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if len(sentence) > 3:
                            font_sentences[font_size].append({
                                'sentence': sentence,
                                'page': line_words[0]['page'],
                                'y0': y0
                            })
    
    return font_sentences

def filter_sentences(font_sentences):
    """조건에 맞는 문장만 필터링합니다."""
    filtered_sentences = []
    
    # 모든 폰트 크기의 문장을 추출 (필터링하지 않음)
    for font_size, sentences in font_sentences.items():
        for sent_info in sentences:
            sentence = sent_info['sentence']
            
            # "..."이 포함된 문장 제외
            if '...' in sentence:
                continue
            
            # 너무 짧은 문장 제외
            if len(sentence) < 3:
                continue
            
            filtered_sentences.append({
                'sentence': sentence,
                'page': sent_info['page'],
                'font_size': font_size,
                'y0': sent_info['y0'],
                'column': sent_info.get('column')  # 컬럼 정보 유지
            })
    
    return filtered_sentences

def create_chunks_from_sentences(sentences, tables):
    """문장들을 chunk로 구성합니다."""
    chunks = []
    
    # 페이지별로 그룹화
    page_sentences = defaultdict(list)
    for sent_info in sentences:
        page_sentences[sent_info['page']].append(sent_info)
    
    # 각 페이지별로 chunk 생성
    for page_num in sorted(page_sentences.keys()):
        page_sents = sorted(page_sentences[page_num], key=lambda x: x['y0'])
        
        # 2단 구조인지 확인 (column 정보가 있는지)
        has_column_info = any('column' in sent for sent in page_sents)
        
        if has_column_info:
            # 2단 구조: 왼쪽 컬럼을 먼저 완전히 처리하고 나서 오른쪽 컬럼 처리
            left_sentences = [s for s in page_sents if s.get('column') == 'left']
            right_sentences = [s for s in page_sents if s.get('column') == 'right']
            
            # 왼쪽 컬럼 먼저 완전히 처리
            left_chunks = process_column_chunks(left_sentences, "left")
            chunks.extend(left_chunks)
            
            # 오른쪽 컬럼 처리
            right_chunks = process_column_chunks(right_sentences, "right")
            chunks.extend(right_chunks)
        else:
            # 1단 구조: 기존 방식 그대로 사용
            # 12pt와 14pt 문장을 제목으로 사용
            title_sentences = [sent for sent in page_sents if sent['font_size'] in [12.0, 14.0]]
            
            for i, title_sent in enumerate(title_sentences):
                title = title_sent['sentence']
                title_y0 = title_sent['y0']
                
                # 다음 제목까지의 내용을 수집
                content_sentences = []
                next_title_y0 = None
                
                if i + 1 < len(title_sentences):
                    next_title_y0 = title_sentences[i + 1]['y0']
                
                # 제목 뒤부터 다음 제목 전까지의 모든 문장을 content로 수집
                for sent in page_sents:
                    if sent['y0'] > title_y0:  # 제목보다 아래에 있는 문장
                        if next_title_y0 is None or sent['y0'] < next_title_y0:  # 다음 제목보다 위에 있는 문장
                            content_sentences.append(sent['sentence'])
                
                # content 구성
                if content_sentences:
                    content = ' '.join(content_sentences)
                else:
                    content = title  # 내용이 없으면 제목을 내용으로 사용
                
                chunk = {
                    'title': title,
                    'content': content,
                    'page': title_sent['page'],
                    'font_size': title_sent['font_size']
                }
                chunks.append(chunk)
    
    # 표 정보를 해당 페이지의 chunk에 추가
    for table in tables:
        table_page = table['page']
        table_text = "TABLE:\n"
        for row in table['data']:
            table_text += " | ".join(row) + "\n"
        
        # 해당 페이지의 chunk에 표 추가
        for chunk in chunks:
            if chunk['page'] == table_page:
                chunk['content'] += "\n\n" + table_text
                break
    
    return chunks

def process_column_chunks(column_sentences, column_name):
    """컬럼의 문장들을 chunk로 처리"""
    chunks = []
    
    # 12pt와 14pt 문장을 제목으로 사용
    title_sentences = [sent for sent in column_sentences if sent['font_size'] in [12.0, 14.0]]
    
    # 폰트 크기별로 그룹화하여 y좌표가 가까운 제목들을 하나로 합치기
    font_groups = defaultdict(list)
    for sent in title_sentences:
        font_groups[sent['font_size']].append(sent)
    
    merged_titles = []
    
    for font_size, font_sentences in font_groups.items():
        # 같은 폰트 크기 내에서 y좌표가 가까운 제목들을 하나로 합치기
        i = 0
        while i < len(font_sentences):
            current_title = font_sentences[i]
            merged_title = current_title.copy()
            
            # 다음 제목들과 y좌표 차이가 20pt 이내인지 확인 (같은 폰트 크기 내에서만)
            j = i + 1
            while j < len(font_sentences):
                next_title = font_sentences[j]
                y_diff = abs(next_title['y0'] - current_title['y0'])
                
                if y_diff <= 20:  # 20pt 이내면 같은 라인으로 간주
                    merged_title['sentence'] += ' ' + next_title['sentence']
                    merged_title['y0'] = min(merged_title['y0'], next_title['y0'])  # 더 작은 y좌표 사용
                    j += 1
                else:
                    break
            
            merged_titles.append(merged_title)
            i = j
    
    # y좌표로 정렬
    merged_titles.sort(key=lambda x: x['y0'])
    
    # 합쳐진 제목들로 chunk 생성
    for i, title_sent in enumerate(merged_titles):
        title = title_sent['sentence']
        title_y0 = title_sent['y0']
        
        # 다음 제목까지의 내용을 수집 (같은 컬럼 내에서)
        content_sentences = []
        next_title_y0 = None
        
        if i + 1 < len(merged_titles):
            next_title_y0 = merged_titles[i + 1]['y0']
        
        # 제목 뒤부터 다음 제목 전까지의 모든 문장을 content로 수집
        for sent in column_sentences:
            if sent['y0'] > title_y0:  # 제목보다 아래에 있는 문장
                if next_title_y0 is None or sent['y0'] < next_title_y0:  # 다음 제목보다 위에 있는 문장
                    content_sentences.append(sent['sentence'])
        
        # content 구성
        if content_sentences:
            content = ' '.join(content_sentences)
        else:
            content = title  # 내용이 없으면 제목을 내용으로 사용
        
        chunk = {
            'title': title,
            'content': content,
            'page': title_sent['page'],
            'font_size': title_sent['font_size'],
            'column': column_name
        }
        chunks.append(chunk)
    
    return chunks

def clean_and_finalize_chunks(chunks):
    """chunk를 정리하고 최종화합니다."""
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
        title = chunk.get('title', '').strip()
        content = chunk.get('content', '').strip()
        
        if not content or len(content) < 20:
            continue
        
        # 특수문자 정리
        title = re.sub(r'\s+', ' ', title)
        content = re.sub(r'\s+', ' ', content)
        
        cleaned_chunk = {
            'id': i,
            'title': title,
            'content': content,
            'page': chunk.get('page', 0),
            'font_size': chunk.get('font_size', 0),
            'column': chunk.get('column')  # 컬럼 정보 유지
        }
        
        cleaned_chunks.append(cleaned_chunk)
    
    return cleaned_chunks

def save_chunks_to_jsonl(chunks, output_path: str):
    """chunk를 JSONL 형식으로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

def main():
    pdf_path = "Manual_EN.pdf"
    output_path = "Manual_EN_chunks.jsonl"
    
    print("PDF에서 문장과 표 추출 중...")
    
    # 문장 추출 (모든 페이지)
    font_sentences = extract_font_sentences(pdf_path)
    
    # 표 추출 (모든 페이지)
    tables = extract_tables_from_pdf(pdf_path)
    
    print("조건에 맞는 문장 필터링 중...")
    filtered_sentences = filter_sentences(font_sentences)
    
    print(f"필터링된 문장 수: {len(filtered_sentences)}")
    print(f"표 수: {len(tables)}")
    
    print("Chunk 생성 중...")
    chunks = create_chunks_from_sentences(filtered_sentences, tables)
    
    print("Chunk 정리 중...")
    cleaned_chunks = clean_and_finalize_chunks(chunks)
    
    print(f"최종 chunk 수: {len(cleaned_chunks)}")
    
    print("JSONL 파일로 저장 중...")
    save_chunks_to_jsonl(cleaned_chunks, output_path)
    
    print(f"완료! {output_path}에 저장되었습니다.")
    
    # 샘플 출력
    print("\n=== 샘플 chunk ===")
    for i, chunk in enumerate(cleaned_chunks[:5]):
        print(f"Chunk {i}:")
        print(f"  제목: {chunk['title']}")
        print(f"  폰트크기: {chunk['font_size']}")
        print(f"  내용: {chunk['content'][:150]}...")
        print(f"  페이지: {chunk['page']}")
        print()

if __name__ == "__main__":
    main() 