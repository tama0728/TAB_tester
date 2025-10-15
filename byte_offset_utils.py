"""
바이트 기반 offset 계산을 위한 유틸리티 함수들
이모지(🚀)와 같은 멀티바이트 문자를 올바르게 처리합니다.
"""

def get_byte_offset(text, char_offset):
    """
    문자 offset을 바이트 offset으로 변환합니다.
    
    Args:
        text (str): 원본 텍스트
        char_offset (int): 문자 기반 offset
    
    Returns:
        int: 바이트 기반 offset
    """
    if char_offset <= 0:
        return 0
    
    # 텍스트를 문자 단위로 슬라이스한 후 UTF-8로 인코딩하여 바이트 길이 계산
    substring = text[:char_offset]
    return len(substring.encode('utf-8'))

def get_char_offset_from_byte(text, byte_offset):
    """
    바이트 offset을 문자 offset으로 변환합니다.
    
    Args:
        text (str): 원본 텍스트
        byte_offset (int): 바이트 기반 offset
    
    Returns:
        int: 문자 기반 offset
    """
    if byte_offset <= 0:
        return 0
    
    # UTF-8 바이트를 문자로 디코딩
    try:
        return len(text.encode('utf-8')[:byte_offset].decode('utf-8'))
    except UnicodeDecodeError:
        # 잘못된 바이트 offset인 경우, 안전하게 처리
        return len(text)

def get_utf16_code_unit_offset(text, char_offset):
    """
    문자 offset을 UTF-16 코드 유닛 offset으로 변환합니다.
    Windows 메모장의 길이 표시 기준과 동일합니다.
    
    Args:
        text (str): 원본 텍스트
        char_offset (int): 문자 기반 offset
    
    Returns:
        int: UTF-16 코드 유닛 기반 offset
    """
    if char_offset <= 0:
        return 0
    
    # 텍스트를 문자 단위로 슬라이스한 후 UTF-16으로 인코딩
    substring = text[:char_offset]
    utf16_bytes = substring.encode('utf-16le')
    
    # UTF-16 코드 유닛 개수 (2바이트 = 1 코드 유닛)
    return len(utf16_bytes) // 2

def get_char_offset_from_utf16_code_unit(text, code_unit_offset):
    """
    UTF-16 코드 유닛 offset을 문자 offset으로 변환합니다.
    
    Args:
        text (str): 원본 텍스트
        code_unit_offset (int): UTF-16 코드 유닛 기반 offset
    
    Returns:
        int: 문자 기반 offset
    """
    if code_unit_offset <= 0:
        return 0
    
    # UTF-16 바이트로 변환
    target_bytes = code_unit_offset * 2
    
    # UTF-16 바이트를 문자로 디코딩
    try:
        utf16_bytes = text.encode('utf-16le')[:target_bytes]
        return len(utf16_bytes.decode('utf-16le'))
    except UnicodeDecodeError:
        # 잘못된 offset인 경우, 안전하게 처리
        return len(text)

def convert_utf16_offset_to_char_range(text, utf16_start, utf16_end):
    """
    UTF-16 코드 유닛 범위를 문자 범위로 변환합니다.
    
    Args:
        text (str): 원본 텍스트
        utf16_start (int): UTF-16 코드 유닛 시작 위치
        utf16_end (int): UTF-16 코드 유닛 끝 위치
    
    Returns:
        tuple: (문자 시작 위치, 문자 끝 위치)
    """
    char_start = get_char_offset_from_utf16_code_unit(text, utf16_start)
    char_end = get_char_offset_from_utf16_code_unit(text, utf16_end)
    return char_start, char_end

def correct_offset_mapping_utf16(text, offset_mapping):
    """
    offset_mapping을 UTF-16 코드 유닛 기반으로 수정합니다.
    Windows 메모장 기준과 동일합니다.
    
    Args:
        text (str): 원본 텍스트
        offset_mapping (list): [(start_char, end_char), ...] 형태의 offset 매핑
    
    Returns:
        list: UTF-16 코드 유닛 기반 offset 매핑
    """
    corrected_mapping = []
    
    for start_char, end_char in offset_mapping:
        if start_char is None or end_char is None:
            corrected_mapping.append((start_char, end_char))
            continue
            
        start_code_unit = get_utf16_code_unit_offset(text, start_char)
        end_code_unit = get_utf16_code_unit_offset(text, end_char)
        
        corrected_mapping.append((start_code_unit, end_code_unit))
    
    return corrected_mapping

def adjust_span_offsets_utf16(text, spans):
    """
    span들의 offset을 UTF-16 코드 유닛 기반으로 조정합니다.
    Windows 메모장 기준과 동일합니다.
    
    Args:
        text (str): 원본 텍스트
        spans (list): span 딕셔너리들의 리스트
    
    Returns:
        list: UTF-16 코드 유닛 기반 offset으로 조정된 spans
    """
    adjusted_spans = []
    
    for span in spans:
        adjusted_span = span.copy()
        
        if 'start_offset' in span:
            adjusted_span['start_offset'] = get_utf16_code_unit_offset(text, span['start_offset'])
        
        if 'end_offset' in span:
            adjusted_span['end_offset'] = get_utf16_code_unit_offset(text, span['end_offset'])
        
        adjusted_spans.append(adjusted_span)
    
    return adjusted_spans

def get_emoji_byte_size(text):
    """
    텍스트에 포함된 이모지들의 바이트 크기를 반환합니다.
    
    Args:
        text (str): 분석할 텍스트
    
    Returns:
        dict: {emoji: byte_size} 형태의 딕셔너리
    """
    emoji_sizes = {}
    
    # 간단한 이모지 감지 (유니코드 범위 기반)
    for char in text:
        if is_emoji(char):
            emoji_sizes[char] = len(char.encode('utf-8'))
    
    return emoji_sizes

def is_emoji(char):
    """
    문자가 이모지인지 확인합니다.
    
    Args:
        char (str): 확인할 문자
    
    Returns:
        bool: 이모지 여부
    """
    # 유니코드 이모지 범위들
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map
        (0x1F1E0, 0x1F1FF),  # Regional indicator symbols
        (0x2600, 0x26FF),    # Miscellaneous symbols
        (0x2700, 0x27BF),    # Dingbats
        (0xFE00, 0xFE0F),    # Variation Selectors
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1F018, 0x1F270),  # Various other emoji ranges
    ]
    
    code_point = ord(char)
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True
    
    return False

def correct_offset_mapping(text, offset_mapping):
    """
    offset_mapping을 바이트 기반으로 수정합니다.
    
    Args:
        text (str): 원본 텍스트
        offset_mapping (list): [(start_char, end_char), ...] 형태의 offset 매핑
    
    Returns:
        list: 바이트 기반 offset 매핑
    """
    corrected_mapping = []
    
    for start_char, end_char in offset_mapping:
        if start_char is None or end_char is None:
            corrected_mapping.append((start_char, end_char))
            continue
            
        start_byte = get_byte_offset(text, start_char)
        end_byte = get_byte_offset(text, end_char)
        
        corrected_mapping.append((start_byte, end_byte))
    
    return corrected_mapping

def adjust_span_offsets(text, spans):
    """
    span들의 offset을 바이트 기반으로 조정합니다.
    
    Args:
        text (str): 원본 텍스트
        spans (list): span 딕셔너리들의 리스트
    
    Returns:
        list: 바이트 기반 offset으로 조정된 spans
    """
    adjusted_spans = []
    
    for span in spans:
        adjusted_span = span.copy()
        
        if 'start_offset' in span:
            adjusted_span['start_offset'] = get_byte_offset(text, span['start_offset'])
        
        if 'end_offset' in span:
            adjusted_span['end_offset'] = get_byte_offset(text, span['end_offset'])
        
        adjusted_spans.append(adjusted_span)
    
    return adjusted_spans

# 테스트 함수
def test_emoji_offset():
    """이모지 offset 계산 테스트"""
    test_text = "Hello 🚀 World 🌍"
    
    print(f"Original text: {test_text}")
    print(f"Text length (chars): {len(test_text)}")
    print(f"Text length (bytes): {len(test_text.encode('utf-8'))}")
    
    # 이모지 위치 찾기
    rocket_pos = test_text.find("🚀")
    earth_pos = test_text.find("🌍")
    
    print(f"🚀 position: {rocket_pos} (char), {get_byte_offset(test_text, rocket_pos)} (byte)")
    print(f"🌍 position: {earth_pos} (char), {get_byte_offset(test_text, earth_pos)} (byte)")
    
    # 이모지 바이트 크기
    emoji_sizes = get_emoji_byte_size(test_text)
    for emoji, size in emoji_sizes.items():
        print(f"{emoji}: {size} bytes")

def test_utf16_code_unit_offset():
    """UTF-16 코드 유닛 offset 계산 테스트 (Windows 메모장 기준)"""
    test_text = "Hello 🚀 World 🌍"
    
    print("\n" + "=" * 60)
    print("UTF-16 코드 유닛 Offset 계산 테스트 (Windows 메모장 기준)")
    print("=" * 60)
    
    print(f"텍스트: '{test_text}'")
    print(f"문자 길이: {len(test_text)}")
    print(f"UTF-16 코드 유닛 길이: {len(test_text.encode('utf-16le')) // 2}")
    print()
    
    # 각 문자별 UTF-16 코드 유닛 위치
    print("각 문자별 UTF-16 코드 유닛 위치:")
    for i, char in enumerate(test_text):
        code_unit_pos = get_utf16_code_unit_offset(test_text, i)
        char_bytes = char.encode('utf-16le')
        code_unit_count = len(char_bytes) // 2
        print(f"문자 {i}: '{char}' -> 코드 유닛 {code_unit_pos} (코드 유닛 개수: {code_unit_count})")
    
    # 이모지 위치
    rocket_pos = test_text.find("🚀")
    earth_pos = test_text.find("🌍")
    
    print(f"\n🚀 이모지:")
    print(f"  문자 위치: {rocket_pos}")
    print(f"  UTF-16 코드 유닛 위치: {get_utf16_code_unit_offset(test_text, rocket_pos)}")
    print(f"  메모장에서 표시되는 길이: 2")
    
    print(f"\n🌍 이모지:")
    print(f"  문자 위치: {earth_pos}")
    print(f"  UTF-16 코드 유닛 위치: {get_utf16_code_unit_offset(test_text, earth_pos)}")
    print(f"  메모장에서 표시되는 길이: 2")

def compare_offset_methods():
    """다양한 offset 계산 방식 비교"""
    
    print("\n" + "=" * 60)
    print("다양한 Offset 계산 방식 비교")
    print("=" * 60)
    
    test_cases = [
        "Hello 🚀 World",
        "안녕하세요 🌍",
        "Test 🎉 emoji 🔥",
        "No emoji text"
    ]
    
    for text in test_cases:
        print(f"\n텍스트: '{text}'")
        print(f"  Python len(): {len(text)} (문자 개수)")
        print(f"  UTF-8 바이트: {len(text.encode('utf-8'))} (실제 저장 공간)")
        print(f"  UTF-16 바이트: {len(text.encode('utf-16le'))} (Windows 저장 공간)")
        print(f"  UTF-16 코드 유닛: {len(text.encode('utf-16le')) // 2} (메모장 표시)")
        
        # 이모지가 있다면 개별 분석
        emojis = [char for char in text if is_emoji(char)]
        if emojis:
            print("  이모지 분석:")
            for emoji in set(emojis):
                pos = text.find(emoji)
                utf16_pos = get_utf16_code_unit_offset(text, pos)
                print(f"    '{emoji}': 문자 {pos} -> 코드 유닛 {utf16_pos}")

if __name__ == "__main__":
    test_emoji_offset()
