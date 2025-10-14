# Python 타입 Annotation 유추 메커니즘 분석

## 목차
1. [개요](#개요)
2. [기본 타입 Annotation 문법](#기본-타입-annotation-문법)
3. [typing 모듈과 제네릭 타입](#typing-모듈과-제네릭-타입)
4. [dataclass를 활용한 타입 시스템](#dataclass를-활용한-타입-시스템)
5. [타입 별칭 (Type Aliases)](#타입-별칭-type-aliases)
6. [타입 체커 제어 (type: ignore)](#타입-체커-제어-type-ignore)
7. [프로젝트별 타입 Annotation 패턴](#프로젝트별-타입-annotation-패턴)
8. [Best Practices](#best-practices)

---

## 개요

Python은 **동적 타입 언어**로, 변수의 타입이 런타임에 결정됩니다. 하지만 **PEP 484**(Python 3.5+)부터 타입 힌트(Type Hints) 시스템이 도입되어 정적 타입 분석이 가능해졌습니다.

### 타입 Annotation의 역할
- **코드 가독성 향상**: 함수의 입출력 타입을 명확히 표시
- **IDE 지원**: 자동완성, 리팩토링, 에러 감지
- **타입 체커**: mypy, pyright 등으로 컴파일 타임 타입 검증
- **문서화**: 타입 정보가 코드 자체에 내장됨

### 런타임 vs 정적 분석
⚠️ **중요**: 타입 annotation은 런타임에 강제되지 않습니다. 이는 **힌트**일 뿐이며, 실제 타입 검증은 별도의 도구(mypy 등)가 수행합니다.

---

## 기본 타입 Annotation 문법

### 1. 함수 파라미터 타입 지정

```python
# evaluation.py:100
def __init__(self, gold_standard_json_file: str, spacy_model = "en_core_web_md"):
    self.nlp = spacy.load(spacy_model)
```

**타입 유추 메커니즘:**
- `gold_standard_json_file: str` → 명시적으로 str 타입 요구
- `spacy_model = "en_core_web_md"` → 기본값이 문자열이므로 암묵적으로 str 추론

### 2. 반환 타입 지정

```python
# data_handling.py:141
def __getitem__(self, idx) -> TrainingExample:
    return self.training_examples[idx]
```

**문법:** `-> ReturnType:` 형태로 반환 타입 명시

### 3. 변수 타입 Annotation

```python
# data_handling.py:150-152
def __init__(self, examples: List[TrainingExample]):
    self.input_ids: torch.Tensor
    self.attention_masks: torch.Tensor
    self.labels: torch.Tensor
```

**특징:**
- 변수 선언과 타입 annotation을 분리 가능
- 나중에 실제 값 할당: `self.input_ids = torch.LongTensor(input_ids)`

### 4. 타입이 없는 파라미터 (암묵적 유추)

```python
# annotate.py:64
def __init__(self, spacy_model="en_core_web_md"):
    self.nlp = spacy.load(spacy_model)
```

**타입 유추:**
- Python 인터프리터는 기본값 `"en_core_web_md"`에서 str 타입 추론
- mypy 같은 타입 체커도 기본값에서 타입 추론
- 하지만 명시적 annotation이 없으면 다른 타입 전달 시 런타임 에러만 발생

---

## typing 모듈과 제네릭 타입

### 1. 기본 제네릭 타입

```python
# evaluation.py:3
from typing import Any, Dict, List, Tuple

# evaluation.py:62
def get_weights(self, text: str, text_spans: List[Tuple[int, int]]):
    return
```

**제네릭 타입 해석:**
- `List[Tuple[int, int]]` → "정수 튜플의 리스트"
- 중첩된 제네릭: `List[T]` 안에 `Tuple[int, int]`
- 각 튜플은 정확히 2개의 정수 포함 (start, end offset)

### 2. typing 모듈의 주요 타입

| 타입 | 설명 | 예제 |
|------|------|------|
| `List[T]` | 요소 타입이 T인 리스트 | `List[str]` |
| `Dict[K, V]` | 키 타입 K, 값 타입 V인 딕셔너리 | `Dict[str, int]` |
| `Tuple[T1, T2]` | 고정 길이 튜플 | `Tuple[int, int]` |
| `Any` | 모든 타입 허용 (타입 체크 무시) | `data: Any` |
| `Optional[T]` | T 또는 None | `Optional[str]` |
| `Union[T1, T2]` | T1 또는 T2 타입 | `Union[int, str]` |

### 3. 복잡한 타입 조합

```python
# evaluation.py:140
def get_entity_recall(self, masked_docs: List[MaskedDocument],
                      include_direct=True, include_quasi=True):
```

**타입 해석:**
- `List[MaskedDocument]` → 사용자 정의 클래스의 리스트
- 나머지 파라미터는 기본값에서 bool 타입 추론

---

## dataclass를 활용한 타입 시스템

### 1. dataclass 기본 사용법

```python
# evaluation.py:18-24
from dataclasses import dataclass

@dataclass
class MaskedDocument:
    doc_id: str
    masked_spans: List[Tuple[int, int]]
```

**dataclass의 장점:**
- 타입이 **필수**: 모든 필드에 타입 annotation 필요
- 자동 생성: `__init__`, `__repr__`, `__eq__` 메서드 자동 생성
- IDE 지원: 타입 정보로 자동완성 강화

**자동 생성된 __init__:**
```python
# dataclass가 자동으로 생성
def __init__(self, doc_id: str, masked_spans: List[Tuple[int, int]]):
    self.doc_id = doc_id
    self.masked_spans = masked_spans
```

### 2. dataclass와 일반 클래스 비교

```python
# evaluation.py:75-86
@dataclass
class AnnotatedEntity:
    entity_id: str
    mentions: List[Tuple[int, int]]
    need_masking: bool
    is_direct: bool
    entity_type: str
    mention_level_masking: List[bool]

    def __post_init__(self):
        if self.is_direct and not self.need_masking:
            raise RuntimeError("Direct identifiers must always be masked")
```

**특징:**
- `__post_init__`: dataclass 초기화 후 추가 검증 수행
- 모든 필드가 타입과 함께 명시되어 있음

### 3. TrainingExample 복잡한 예제

```python
# data_handling.py:47-53
@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList
    identifier_types: IntList
    offsets: IntList
```

**타입 별칭과 결합:**
- `IntList`는 사용자 정의 타입 별칭 (아래 섹션 참조)
- 코드 재사용성과 가독성 향상

---

## 타입 별칭 (Type Aliases)

### 1. 간단한 타입 별칭

```python
# data_handling.py:26-27
IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids
```

**사용 이유:**
- 복잡한 타입을 간단한 이름으로 재사용
- 의미를 명확히 전달: `IntList` → "정수 리스트"가 아니라 "토큰 ID 리스트"
- 타입 변경 시 한 곳만 수정

### 2. 타입 별칭 활용 예제

```python
# data_handling.py:49-51
@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList
```

**Without Type Alias (타입 별칭 없이):**
```python
@dataclass
class TrainingExample:
    input_ids: List[int]
    attention_masks: List[int]
    labels: List[int]
```

**비교:**
- 타입 별칭 사용 → 간결하고 의미 명확
- 반복 제거 → DRY 원칙 준수

### 3. 중첩된 타입 별칭

```python
# data_handling.py:155-159
def __init__(self, examples: List[TrainingExample]):
    input_ids: IntListList = []
    masks: IntListList = []
    labels: IntListList = []
```

**타입 해석:**
- `IntListList` = `List[List[int]]`
- 2차원 정수 배열 (배치 데이터 구조)

---

## 타입 체커 제어 (type: ignore)

### 1. type: ignore의 목적

```python
# annotate.py:2
from spacy.tokens import Span  #type: ignore
```

**사용 이유:**
- spaCy 라이브러리가 타입 스텁(type stub)을 제공하지 않음
- mypy/pyright가 경고를 발생시키는 것을 방지

### 2. 코드 내부에서 type: ignore 사용

```python
# annotate.py:144
new_ents.append((ent.start, ent.end+1, spacy.symbols.MONEY))  #type: ignore
```

**왜 필요한가?**
- `spacy.symbols.MONEY`의 타입을 mypy가 추론하지 못함
- 타입 불일치 경고를 억제하되, 개발자는 타입이 올바르다는 것을 확신

### 3. type: ignore 남용 주의

```python
# annotate.py:154-156
if (doc[ent.end-1].text=="'s" and ent.label==spacy.symbols.PERSON):  #type: ignore
    new_ents.append((ent.start, ent.end-1, spacy.symbols.PERSON))  #type: ignore
```

**Best Practice:**
- 외부 라이브러리 타입 불일치에만 사용
- 자체 코드에서는 타입 annotation 개선이 우선
- 주석으로 이유 설명 추가 권장

---

## 프로젝트별 타입 Annotation 패턴

### 1. evaluation.py - 고급 타입 시스템

**특징:**
- 완전한 타입 annotation
- dataclass 적극 활용
- 추상 클래스와 타입 시스템 결합

```python
# evaluation.py:58-72
class TokenWeighting:
    @abc.abstractmethod
    def get_weights(self, text: str, text_spans: List[Tuple[int,int]]):
        return
```

**패턴 분석:**
- 추상 메서드에도 타입 힌트 제공
- 인터페이스 명확성 향상
- 구현 클래스에서 타입 일치 강제

### 2. data_handling.py - 타입 별칭 중심

**특징:**
- 타입 별칭으로 코드 간결화
- dataclass와 타입 별칭 결합
- 복잡한 데이터 구조를 명확히 표현

```python
# data_handling.py:145-154
class TrainingBatch:
    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        self.identifier_types: List
        self.offsets: List
```

**패턴 분석:**
- 속성 타입을 먼저 선언 후 나중에 할당
- PyTorch Tensor 타입 명시
- 일부는 제네릭 없이 `List`만 사용 (간결함 우선)

### 3. annotate.py - 부분적 타입 힌트

**특징:**
- 레거시 코드 스타일
- 기본값 위주의 암묵적 타입 유추
- type: ignore로 외부 라이브러리 처리

```python
# annotate.py:67-69
def annotate_all(self, texts):
    for doc in self.nlp.pipe(texts):
        yield self._annotate(doc)
```

**개선 가능 버전:**
```python
from typing import Iterator
import spacy

def annotate_all(self, texts: List[str]) -> Iterator[spacy.tokens.Doc]:
    for doc in self.nlp.pipe(texts):
        yield self._annotate(doc)
```

### 4. longformer_model.py - 프레임워크 의존형

**특징:**
- PyTorch 모델은 타입 힌트 최소화
- 프레임워크가 타입 추론 담당
- 동적 그래프 특성으로 타입 힌트 어려움

```python
# longformer_model.py:22-29
def __init__(self, model, num_labels):
    super().__init__()
    self._bert = LongformerModel.from_pretrained(model)
    self.classifier = nn.Linear(768, num_labels)
```

**개선 가능 버전:**
```python
def __init__(self, model: str, num_labels: int) -> None:
    super().__init__()
    self._bert = LongformerModel.from_pretrained(model)
    self.classifier = nn.Linear(768, num_labels)
```

---

## Best Practices

### 1. 타입 Annotation 작성 우선순위

**높음 (반드시 작성):**
- ✅ 공개 API 함수/메서드
- ✅ 복잡한 데이터 구조 (dataclass)
- ✅ 라이브러리 인터페이스

**중간 (권장):**
- ⚠️ 내부 헬퍼 함수
- ⚠️ 간단한 변수 (추론 가능한 경우 생략 가능)

**낮음 (선택):**
- ⭕ 명확한 기본값이 있는 파라미터
- ⭕ 로컬 변수 (IDE가 추론 가능)

### 2. 점진적 타입 도입 전략

```python
# Step 1: 함수 시그니처부터
def process_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    pass

# Step 2: 복잡한 타입은 별칭으로
DataRecord = Dict[str, Any]
def process_data(data: List[DataRecord]) -> pd.DataFrame:
    pass

# Step 3: dataclass로 구조화
@dataclass
class DataRecord:
    id: str
    value: float
    metadata: Dict[str, Any]

def process_data(data: List[DataRecord]) -> pd.DataFrame:
    pass
```

### 3. 타입 체커 활용

**mypy 실행:**
```bash
# 프로젝트 전체 타입 체크
mypy .

# 특정 파일만
mypy evaluation.py
```

**pyright (VSCode 기본):**
```bash
pyright evaluation.py
```

### 4. 타입 Annotation의 한계 인지

**런타임 검증 없음:**
```python
def add(a: int, b: int) -> int:
    return a + b

# 타입 힌트 무시하고 실행됨 (런타임 에러 없음)
result = add("hello", "world")  # "helloworld"
```

**해결책: pydantic 사용**
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# 런타임에 타입 검증
user = User(name="Alice", age="30")  # ValidationError!
```

### 5. 프로젝트 타입 일관성 유지

**좋은 예:**
```python
# 모든 파일에서 일관된 타입 시스템
from typing import List, Dict, Tuple

def func1(data: List[str]) -> Dict[str, int]:
    pass

def func2(records: List[Dict[str, Any]]) -> None:
    pass
```

**나쁜 예:**
```python
# 파일마다 다른 스타일
def func1(data):  # 타입 없음
    pass

def func2(records: list) -> None:  # 제네릭 없음
    pass

def func3(items: List[Dict[str, Any]]) -> dict:  # 혼용
    pass
```

---

## 요약

### Python 타입 유추 메커니즘

1. **명시적 Annotation**: `param: Type` 형태로 직접 지정
2. **typing 모듈**: 제네릭 타입으로 복잡한 구조 표현
3. **dataclass**: 타입 기반 데이터 클래스 자동 생성
4. **타입 별칭**: 재사용성과 가독성 향상
5. **암묵적 유추**: 기본값이나 런타임 동작에서 추론
6. **타입 체커 제어**: `type: ignore`로 외부 라이브러리 처리

### 타입 Annotation의 가치

- **개발 생산성**: IDE 자동완성, 리팩토링 지원
- **버그 예방**: 컴파일 타임에 타입 에러 감지
- **문서화**: 코드가 스스로 설명하는 구조
- **유지보수성**: 대규모 프로젝트에서 타입 안정성 확보

### 프로젝트 적용 현황

| 파일 | 타입 완성도 | 특징 |
|------|------------|------|
| evaluation.py | ⭐⭐⭐⭐⭐ | 완전한 타입 시스템, dataclass 활용 |
| data_handling.py | ⭐⭐⭐⭐⭐ | 타입 별칭 중심, PyTorch 통합 |
| annotate.py | ⭐⭐⭐ | 부분적 타입 힌트, type: ignore 사용 |
| longformer_model.py | ⭐⭐ | 최소한의 타입 힌트, 프레임워크 의존 |

---

**작성일**: 2025-10-01
**분석 대상**: text-anonymization-benchmark 프로젝트
**분석 도구**: Sequential Thinking MCP
