import os
import io
import json
import asyncio
import re
import uuid
import base64
import numpy as np
import torch
import soundfile as sf
import resampy
from datetime import datetime
from pathlib import Path
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
from PyPDF2 import PdfReader
from docx import Document
import pdfplumber  # Для более надёжного чтения PDF
from fastapi import File, UploadFile, Form
from typing import Optional
import re  # Добавьте импорт в начало файла, если не
import librosa

# Импорты для FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict

# Ваши импорты для AI-моделей
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =======================
# Инициализация FastAPI и настройка
# =======================
app = FastAPI()
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)  # Создаем папку для отчетов, если ее нет
INTERVIEW_DATA_FILE = "interview_data.json"
chat_history: Dict[str, list] = {}  # История чата для каждой сессии по interview_id

# =======================
# Pydantic модели
# =======================
class VacancyItem(BaseModel):
    key: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$", description="Ключ: только латиница, цифры, _ и -")
    title: str
    requirements: str
    questions: List[str]
    script: str
    criteria_weights: str = Field("{}", description="JSON-строка с весами критериев, например: {\"technical_skills\": 50, \"communication\": 30, \"cases\": 20}")

class LLMAnalysisResponse(BaseModel):
    """Структура ответа от LLM, включающая публичный ответ и внутренний анализ."""
    assistant_response: str
    internal_analysis: str

# =======================
# Глобальные переменные и работа с данными
# =======================
def read_data() -> dict:
    """Читает и возвращает все данные из JSON файла."""
    if not os.path.exists(INTERVIEW_DATA_FILE):
        return {}
    try:
        with open(INTERVIEW_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Предупреждение: не удалось прочитать или декодировать {INTERVIEW_DATA_FILE}. Возвращен пустой словарь.")
        return {}

def write_data(data: dict):
    """Записывает данные в JSON файл."""
    with open(INTERVIEW_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =======================
# Настройки и загрузка моделей
# =======================
print("Загрузка AI моделей...")

# 1. TTS Model
print("Загрузка модели TTS (XTTS v2)...")
tts_config = XttsConfig()
tts_config.load_json("../tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
tts_model = Xtts.init_from_config(tts_config)
tts_model.load_checkpoint(tts_config, checkpoint_dir="../tts/tts_models--multilingual--multi-dataset--xtts_v2/", use_deepspeed=True)
tts_model.cuda()
speaker_name = "Luis Moray"
gpt_cond_latent, speaker_embedding = tts_model.speaker_manager.speakers[speaker_name].values()

# 2. LLM Model
print("Загрузка LLM (OpenAI)...")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base="https://api.proxyapi.ru/openai/v1", model_name="gpt-4o-mini", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["chat_history", "emotion", "pauses_info", "structure_analysis", "vacancy", "requirements", "questions_list", "interview_script", "criteria_weights"],
    template="""Ты — AI-ассистент, проводящий техническое собеседование на вакансию {vacancy}.
Твой стиль общения: профессиональный, но дружелюбный и поддерживающий. Избегай канцеляризмов.
Твоя главная задача — гибко управлять ходом беседы, а не слепо следовать списку вопросов.

**Требования к кандидату:**
{requirements}

**Веса критериев для оценки (используй для расчёта процента соответствия):**
{criteria_weights}

**Инструкции:**
1. **Используй требования:** Внимательно изучи требования к кандидату. Твои вопросы должны помогать раскрыть, соответствует ли кандидат этим требованиям.
2. **Следуй скрипту:** У тебя есть общий план ведения диалога: {interview_script}.
3. **Используй вопросы как ориентир:** Вот примерный список тем для обсуждения: {questions_list}. Не задавай их подряд. Выбирай следующий вопрос или тему на основе ответа кандидата и требований к вакансии.
4. **Адаптируйся:** Если ответ кандидата глубокий и интересный, задай уточняющие вопросы по этой же теме. Если ответ неуверенный, можешь задать наводящий вопрос или плавно перейти к другой теме.
5. **Анализируй soft skills:** Обращай внимание на следующие сигналы от кандидата:
    - Эмоциональная окраска последнего ответа: {emotion}.
    - Анализ пауз в речи: {pauses_info}. Используй для оценки уверенности (длинные паузы — неуверенность, короткие — уверенность).
    - Логическая структура ответа: {structure_analysis}.
    Используй эту информацию, чтобы понять состояние кандидата (уверенность, замешательство) и скорректировать свой следующий вопрос. Не нужно комментировать эти сигналы вслух.
6. **Веди диалог естественно:** Не превращай собеседование в допрос. Твои ответы должны быть логичным продолжением диалога.
7. **Автоматический анализ в internal_analysis:** 
    - Сопоставь ответ с требованиями через анализ ключевых фраз, навыков, опыта. Выдели подтверждённые/неподтверждённые пункты.
    - Выяви противоречия (расхождение в стаже, несоответствия) и "красные флаги" (шаблонные ответы, уклонение).
    - Рассчитай процент соответствия: Оцени каждый критерий по весам, суммируй (0-100%).
    - **ВАЖНО**: `internal_analysis` должен быть СТРОКОЙ, а не словарем. Сериализуй данные в текст.

**История диалога:**
{chat_history}

**Твоя задача:**
Проанализировав всю информацию выше, сформулируй JSON-объект с двумя ключами:
1. `"assistant_response"`: Твой следующий вопрос или комментарий кандидату (строка).
2. `"internal_analysis"`: Внутренний анализ ответа кандидата (СТРОКА), включающий:
   - Подтверждённые навыки/опыт (например, "Подтверждено: знание Python").
   - Неподтверждённые навыки/опыт (например, "Неподтверждено: опыт с SQL").
   - Противоречия и "красные флаги" (например, "Красный флаг: уклонение от вопроса о стаже").
   - Процент соответствия (например, "Процент: 70% (technical_skills: 80%*0.5 + communication: 60%*0.3 + cases: 50%*0.2)").
   - Оценку ясности, уверенности и структурированности ответа.

**Пример выхода (JSON):**
{{
  "assistant_response": "Спасибо за рассказ! Уточните, пожалуйста, какую систему управления состоянием вы использовали?",
  "internal_analysis": "Подтверждено: опыт с React. Неподтверждено: знание Redux. Красные флаги: уклонение от деталей. Процент: 70% (technical_skills: 80%*0.5 + communication: 60%*0.3 + cases: 50%*0.2). Ответ структурирован, но неполный."
}}

**ВАЖНО:** Верни ТОЛЬКО валидный JSON, где `internal_analysis` — строка.

JSON_OUTPUT:
"""
)

final_report_prompt_template = PromptTemplate(
    input_variables=["dialogue_history", "requirements", "criteria_weights"],
    template="""Ты — AI-ассистент, анализирующий собеседование. На основе диалога ({dialogue_history}) и требований вакансии ({requirements}) с весами критериев ({criteria_weights}) создай финальный отчёт в JSON-формате.

**Инструкции:**
- Проанализируй диалог, сопоставь ответы с требованиями, учти веса критериев.
- Верни JSON-объект с полями:
  - "match_percentage": Процент соответствия (0-100%).
  - "competencies_breakdown": Текст с сильными сторонами и пробелами (например, "Сильные стороны: Знание Python. Пробелы: Нет опыта с SQL").
  - "recommendation": "на следующий этап", "отказ" или "требуется уточнение".
  - "notification_text": Короткий текст обратной связи для кандидата (например, "Ваш уровень Python хорош, но нужно подтянуть SQL").

**Пример JSON:**
{{
  "match_percentage": 75,
  "competencies_breakdown": "Сильные стороны: Опыт с Django, хорошая коммуникация. Пробелы: Недостаточно знаний SQL.",
  "recommendation": "на следующий этап",
  "notification_text": "Ваш уровень Python соответствует требованиям, но рекомендуем углубить знания SQL."
}}

**ВАЖНО:** Верни ТОЛЬКО валидный JSON, без лишнего текста.

JSON_OUTPUT:
{dialogue_history}
"""
)

structure_prompt_template = PromptTemplate(
    input_variables=["user_text"],
    template="Проанализируй следующий текст ответа кандидата: \"{user_text}\". Оцени его логическую структуру по следующим критериям: последовательность изложения, наличие четких аргументов, связность повествования. Дай краткую оценку в 1-2 предложения для внутреннего анализа soft skills ассистентом. Пример: 'Ответ структурирован хорошо, кандидат приводит последовательные аргументы.' или 'Повествование немного сумбурное, кандидат перескакивает с темы на тему.'"
)

resume_analysis_prompt_template = PromptTemplate(
    input_variables=["resume_text", "vacancy", "requirements", "questions_list", "criteria_weights"],
    template="""Ты — AI-аналитик резюме, проводящий глубокий анализ кандидата для вакансии {vacancy}.

**Полный текст резюме:**
{resume_text}

**Требования к вакансии:**
{requirements}

**Вопросы/темы для оценки (используй как ориентир для навыков):**
{questions_list}

**Веса критериев для расчёта процента (сумма весов = 100):**
{criteria_weights}

**Инструкции для анализа:**
- **Извлеки ключевые элементы из резюме:** Навыки (технологии, инструменты), стаж (общий и по релевантным позициям), образование, достижения/проекты, личные качества.
- **Сопоставь с требованиями:** Выдели подтверждённые навыки/опыт (с примерами из резюме), неподтверждённые (что отсутствует), противоречия (например, стаж не соответствует заявленным навыкам).
- **Красные флаги:** Выяви потенциальные проблемы (уклончивые формулировки, пробелы в опыте, несоответствия дат, отсутствие релевантных достижений). Приведи 2-3 примера.
- **Расчёт процента:** Оцени каждый критерий (0-100%) по весам, суммируй общий процент. Учитывай стаж, навыки, образование.
- **Разбивка по компетенциям:** Подробно опиши сильные стороны (что соответствует), слабые стороны (пробелы), советы по улучшению (что изучить/доработать).
- **Рекомендация:** "на следующий этап" (если >70%), "отказ" (если <50%), "требуется уточнение" (иначе). Обоснуй.
- **Отзыв:** Персонализированный текст для кандидата (2-4 предложения, мотивирующий, с конкретными советами).

**Верни чистый JSON без markdown-обёрток (типа ```json
{{
  "match_percentage": 75,  // Общий процент (0-100)
  "key_skills_from_resume": ["Python", "C++", "JavaScript", "OpenCV", "PyTorch", "TensorFlow", "Git"],  // Ключевые навыки из резюме (массив строк)
  "experience_summary": "Стаж: 3 года в backend. Образование: Высшее (МГУ).",  // Краткий обзор стажа и образования
  "confirmed_skills": "Подтверждено: Python (3 года опыта), Git. Примеры: Проекты на GitHub.",  // Подтверждённые навыки с примерами
  "unconfirmed_skills": "Неподтверждено: REST API, английский. Отсутствует в резюме.",  // Неподтверждённые с объяснением
  "red_flags": ["Пробел в опыте 2020-2022 гг.", "Нет quantifiable достижений (метрик)."],  // Массив красных флагов
  "competencies_breakdown": "Technical skills (50%): 80% (хорошо с Python, но слаб SQL). Communication (30%): 60%. Cases (20%): 50%. Советы: Изучить SQL, добавить метрики в резюме.",  // Расширенная разбивка + советы
  "recommendation": "на следующий этап",  // Обоснованная рекомендация
  "notification_text": "Ваш опыт с Python впечатляет, но рекомендуем углубить знания SQL и добавить quantifiable достижения в резюме. Ждём на следующем этапе!"  // Отзыв (2-4 предложения)
}}

**ВАЖНО:** Только валидный JSON. Будь объективным, используй факты из резюме. Если резюме неполное, отметь это.

JSON_OUTPUT:
"""
)

# 3. ASR and Emotion Recognition Models
print("Загрузка моделей ASR (Whisper) и Emotion Recognition (Hubert)...")
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = WhisperModel("large-v3", device=DEVICE, compute_type="float16")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
emotion_model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
emotion_model.to(DEVICE)
print("✅ Все модели успешно загружены.")

# =======================
# Функции-помощники
# =======================
def split_text_into_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Разделяет текст на фрагменты по несколько предложений для стриминга TTS."""
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        if not sentence: continue
        if current_chunk: current_chunk += " "
        current_chunk += sentence
        if (i + 1) % sentences_per_chunk == 0 or (i + 1) == len(sentences):
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk: chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk]

def predict_emotion(audio_array: np.ndarray) -> str:
    """Предсказывает эмоцию по аудиоданным."""
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    inputs = feature_extractor(audio_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True, max_length=16000 * 10, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(inputs['input_values']).logits
    predictions = torch.argmax(logits, dim=-1)
    return num2emotion[predictions.item()]

def analyze_structure(user_text: str) -> str:
    """Анализирует логическую структуру текста с помощью LLM."""
    prompt = structure_prompt_template.format(user_text=user_text)
    return llm.invoke(prompt).content

def generate_and_analyze_response(interview_id: str, user_text: str, emotion: str, pauses_info: str, structure_analysis: str, vacancy_data: dict) -> LLMAnalysisResponse:
    """Генерирует ответ ассистента и внутренний анализ, возвращая структурированный объект."""
    global chat_history
    current_history = chat_history.get(interview_id, [])
    current_history.append(("user", user_text))
    
    history_str = "\n".join([f"{'Кандидат' if role == 'user' else 'Ассистент'}: {content}" for role, content in current_history])
    
    prompt = prompt_template.format(
        chat_history=history_str, 
        emotion=emotion, 
        pauses_info=pauses_info,
        structure_analysis=structure_analysis, 
        vacancy=vacancy_data['title'],
        requirements=vacancy_data['requirements'],
        questions_list=vacancy_data['questions_list'],
        interview_script=vacancy_data['script'],
        criteria_weights=vacancy_data['criteria_weights']
    )
    
    try:
        response_content = llm.invoke(prompt).content.strip()
        print(f"LLM response in generate_and_analyze_response: {response_content}")  # Логируем ответ LLM
        
        if not response_content:
            raise ValueError("LLM вернула пустой ответ")
        
        response_data = json.loads(response_content)
        
        # Если internal_analysis вернулся как словарь, преобразуем в строку
        if isinstance(response_data.get('internal_analysis'), dict):
            response_data['internal_analysis'] = json.dumps(response_data['internal_analysis'], ensure_ascii=False)
        
        analysis_response = LLMAnalysisResponse(**response_data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"ОШИБКА: Не удалось распарсить JSON в generate_and_analyze_response. Ошибка: {e}. Ответ LLM: {response_content}")
        analysis_response = LLMAnalysisResponse(
            assistant_response="Прошу прощения, у меня возникла небольшая техническая проблема. Не могли бы вы повторить свой последний ответ?",
            internal_analysis=f"КРИТИЧЕСКАЯ ОШИБКА: LLM не вернула валидный JSON. Ответ был: {response_content}"
        )
    
    current_history.append(("assistant", analysis_response.assistant_response))
    chat_history[interview_id] = current_history
    return analysis_response

def generate_final_report(report_path: Path, vacancy_data: dict):
    """Генерирует финальный отчёт с помощью LLM на основе диалога."""
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    
    dialogue_history = "\n".join([f"Ассистент: {turn.get('assistant_text', '')}\nКандидат: {turn.get('user_text', '')}\nАнализ: {turn.get('internal_analysis', '')}" for turn in report_data["dialogue"]])
    
    # Проверяем наличие всех необходимых полей
    requirements = vacancy_data.get('requirements', 'Требования не указаны.')
    criteria_weights = vacancy_data.get('criteria_weights', '{}')
    
    # Упрощённый промпт для надёжного JSON
    prompt = final_report_prompt_template.format(
        dialogue_history=dialogue_history[:5000],  # Ограничиваем длину, чтобы не превысить лимит токенов
        requirements=requirements,
        criteria_weights=criteria_weights
    )
    
    try:
        response_content = llm.invoke(prompt).content.strip()
        print(f"LLM response: {response_content}")  # Логируем ответ для диагностики
        
        if not response_content:
            raise ValueError("LLM вернула пустой ответ")
        
        final_report = json.loads(response_content)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"ОШИБКА: Не удалось распарсить JSON для финального отчёта. Ошибка: {e}. Ответ LLM: {response_content}")
        final_report = {
            "match_percentage": 0,
            "competencies_breakdown": "Ошибка генерации: LLM вернула некорректный или пустой ответ.",
            "recommendation": "требуется уточнение",
            "notification_text": "Извините, произошла ошибка при формировании отчёта. Пожалуйста, обратитесь к администратору."
        }
    
    report_data["final_report"] = final_report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    return final_report

def extract_text_from_resume(file_content: bytes, file_extension: str) -> str:
    """Извлекает текст из файла резюме (txt, pdf, doc, docx)."""
    try:
        if file_extension.lower() == 'txt':
            return file_content.decode('utf-8')
        elif file_extension.lower() in ['pdf', 'pdf.txt']:  # PDF
            # Используем pdfplumber для лучшей обработки
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            return text if text else 'Текст не извлечён из PDF.'
        elif file_extension.lower() == 'docx':
            doc = Document(io.BytesIO(file_content))
            text = '\n'.join(para.text for para in doc.paragraphs)
            return text
        elif file_extension.lower() == 'doc':
            # Для .doc используем PyPDF2 как fallback (или конвертируйте в DOCX заранее)
            # Примечание: .doc сложнее, рекомендую конвертировать в DOCX; здесь заглушка
            return 'Поддержка .doc ограничена. Рекомендуем DOCX или PDF.'
        else:
            return 'Формат не поддерживается.'
    except Exception as e:
        print(f"Ошибка извлечения текста: {e}")
        return f'Ошибка чтения файла: {str(e)}'

def analyze_pauses(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Анализирует паузы (тишину) в аудио ответа кандидата."""
    try:
        # Нормализуем аудио, если нужно (убедимся, что в диапазоне [-1, 1])
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Обнаруживаем интервалы тишины (паузы) с порогом -20 dB (тишина ниже этого уровня)
        intervals = librosa.effects.split(audio_array, top_db=20)
        
        # Рассчитываем общую длительность пауз (в секундах)
        total_pause_duration = sum((end - start) / sample_rate for start, end in intervals)
        num_pauses = len(intervals)
        avg_pause_duration = total_pause_duration / num_pauses if num_pauses > 0 else 0
        
        analysis = f"Общая длительность пауз: {total_pause_duration:.2f} сек. ({num_pauses} пауз). Средняя пауза: {avg_pause_duration:.2f} сек."
        
        # Интерпретация для LLM (сигналы неуверенности)
        if total_pause_duration > 5.0:
            analysis += " (Длинные паузы — возможная неуверенность или раздумья)."
        elif total_pause_duration < 1.0:
            analysis += " (Короткие паузы — уверенная речь)."
        
        return analysis
    except Exception as e:
        print(f"Ошибка анализа пауз: {e}")
        return "Анализ пауз недоступен (техническая ошибка)."
    
# =======================
# API Эндпоинты для Админ-панели и Архива
# =======================
@app.get("/admin")
async def get_admin_panel():
    try:
        with open("templates/admin.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Ошибка: файл templates/admin.html не найден.</h1>", status_code=404)

@app.get("/archive")
async def get_archive_page():
    try:
        with open("templates/archive.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл шаблона архива не найден.")
    
@app.get("/analyze")
async def get_analyze_page():
    """Возвращает страницу анализа резюме."""
    try:
        with open("templates/analyze.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл templates/analyze.html не найден.")

@app.get("/api/vacancies")
async def get_vacancies_list():
    """Возвращает список ключей и названий всех вакансий."""
    data = read_data()
    return [{"key": key, "title": value.get("title", "Без названия")} for key, value in data.items()]

@app.get("/api/vacancies/{key}")
async def get_vacancy_details(key: str):
    """Возвращает полную информацию по одной вакансии."""
    data = read_data()
    if key not in data:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")
    return data[key]

@app.post("/api/vacancies", status_code=201)
async def create_vacancy(item: VacancyItem):
    """Создает новую вакансию."""
    data = read_data()
    key = item.key
    if key in data:
        raise HTTPException(status_code=409, detail="Вакансия с таким ключом уже существует")
    
    data[key] = item.dict(exclude={'key'})
    write_data(data)
    return {"message": f"Вакансия '{key}' успешно создана"}

@app.put("/api/vacancies/{original_key}")
async def update_vacancy(original_key: str, item: VacancyItem):
    """Обновляет существующую вакансию."""
    data = read_data()
    if original_key not in data:
        raise HTTPException(status_code=404, detail="Редактируемая вакансия не найдена")
    
    new_key = item.key
    if original_key != new_key:
        if new_key in data:
            raise HTTPException(status_code=409, detail=f"Ключ '{new_key}' уже занят другой вакансией")
        del data[original_key]
    
    data[new_key] = item.dict(exclude={'key'})
    write_data(data)
    return {"message": f"Вакансия '{new_key}' успешно обновлена"}

@app.delete("/api/vacancies/{key}")
async def delete_vacancy(key: str):
    """Удаляет вакансию."""
    data = read_data()
    if key not in data:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")
    
    del data[key]
    write_data(data)
    return {"message": f"Вакансия '{key}' успешно удалена"}

@app.get("/api/reports")
async def get_reports_list():
    """Сканирует папку reports и возвращает метаданные каждого отчета."""
    reports_meta = []
    for report_file in REPORTS_DIR.glob("*.json"):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                reports_meta.append({
                    "id": data.get("interview_id"),
                    "title": data.get("vacancy_title"),
                    "date": data.get("date")
                })
        except Exception as e:
            print(f"Не удалось прочитать файл отчета {report_file}: {e}")
    return reports_meta

@app.get("/api/reports/{interview_id}")
async def get_report_content(interview_id: str):
    """Возвращает полное содержимое указанного отчета."""
    report_path = REPORTS_DIR / f"{interview_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Отчет не найден.")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/reports/{interview_id}/generate")
async def generate_report(interview_id: str):
    """Генерирует финальный отчёт для собеседования."""
    report_path = REPORTS_DIR / f"{interview_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Отчет не найден.")
    
    # Находим вакансию по ID отчёта
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    vacancy_title = report_data.get("vacancy_title", "")
    interview_data = read_data()
    vacancy_key = [key for key, value in interview_data.items() if value["title"] == vacancy_title]
    vacancy_key = vacancy_key[0] if vacancy_key else "default"
    
    default_vacancy = {
        "title": "Собеседование по умолчанию",
        "requirements": "Требования не указаны.",
        "questions_list": "Пожалуйста, администратор должен настроить вакансии.",
        "script": "Сообщить пользователю об ошибке конфигурации.",
        "criteria_weights": "{}"
    }
    vacancy_data = interview_data.get(vacancy_key, default_vacancy)
    
    print(f"Vacancy data for report generation: {vacancy_data}")  # Логируем данные вакансии
    
    final_report = generate_final_report(report_path, vacancy_data)
    return final_report

class ResumeAnalysisRequest(BaseModel):
    vacancy_key: str  # Ключ вакансии

@app.post("/api/resume/analyze")
async def analyze_resume(
    file: UploadFile = File(..., description="Файл резюме (txt, pdf, doc, docx)"),
    vacancy_key: str = Form(..., description="Ключ вакансии")
):
    """Анализирует резюме кандидата и сравнивает с вакансией."""
    if not vacancy_key:
        raise HTTPException(status_code=400, detail="Укажите ключ вакансии")
    
    # Проверяем файл
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не загружен")
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['txt', 'pdf', 'doc', 'docx']:
        raise HTTPException(status_code=400, detail="Поддерживаемые форматы: txt, pdf, doc, docx")
    
    # Читаем файл
    file_content = await file.read()
    
    # Извлекаем текст
    resume_text = extract_text_from_resume(file_content, file_extension)
    if not resume_text or "Ошибка" in resume_text:
        raise HTTPException(status_code=400, detail=f"Не удалось извлечь текст: {resume_text}")
    
    # Получаем данные вакансии
    interview_data = read_data()
    if vacancy_key not in interview_data:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")
    
    vacancy_data = interview_data[vacancy_key]
    questions_list = "\n".join(vacancy_data.get("questions", []))
    
    # Генерируем анализ через LLM
    prompt = resume_analysis_prompt_template.format(
        resume_text=resume_text,
        vacancy=vacancy_data["title"],
        requirements=vacancy_data.get("requirements", "Требования не указаны."),
        questions_list=questions_list,
        criteria_weights=vacancy_data.get("criteria_weights", "{}")
    )
    
    try:
        response_content = llm.invoke(prompt).content.strip()
        print(f"LLM resume analysis response: {response_content}")
        
        if not response_content:
            raise ValueError("LLM вернула пустой ответ")
        
        # Очистка от markdown-обёрток (````json` и ````)
        response_content = re.sub(r'^```json\s*|\s*```$', '', response_content.strip(), flags=re.MULTILINE)
        response_content = response_content.strip()  # Удаляем лишние пробелы/переносы
        
        print(f"Cleaned response_content: {response_content}")  # Лог для диагностики
        
        analysis_result = json.loads(response_content)
        
        # Дефолты для новых полей, если LLM не вернула
        analysis_result.setdefault("key_skills_from_resume", [])
        analysis_result.setdefault("experience_summary", "Не указано")
        analysis_result.setdefault("confirmed_skills", "Нет подтверждённых навыков")
        analysis_result.setdefault("unconfirmed_skills", "Все навыки неподтверждены")
        analysis_result.setdefault("red_flags", [])
        analysis_result.setdefault("competencies_breakdown", "Нет разбивки")
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Ошибка анализа резюме: {e}. Ответ LLM: {response_content}")
        analysis_result = {
            "match_percentage": 0,
            "key_skills_from_resume": [],
            "experience_summary": "Ошибка анализа",
            "confirmed_skills": "",
            "unconfirmed_skills": "",
            "red_flags": ["Ошибка генерации анализа"],
            "competencies_breakdown": "Ошибка: некорректный ответ LLM.",
            "recommendation": "требуется уточнение",
            "notification_text": "Произошла ошибка при анализе резюме."
        }
    
    return analysis_result

# =======================
# Основные эндпоинты приложения
# =======================
@app.get("/")
async def get():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, vacancy: str = "junior"):
    global chat_history
    await websocket.accept()
    
    # 1. Инициализация сессии
    interview_id = str(uuid.uuid4())
    chat_history[interview_id] = []
    
    # 2. Настройка вакансии
    interview_data = read_data()
    vacancy_key = vacancy.lower().strip()
    
    if vacancy_key not in interview_data:
        vacancy_key = next(iter(interview_data)) if interview_data else "default"

    selected_vacancy_info = interview_data.get(vacancy_key, {
        "title": "Собеседование по умолчанию",
        "requirements": "Требования не указаны.",
        "questions": ["Пожалуйста, администратор должен настроить вакансии в панели управления."],
        "script": "Сообщить пользователю об ошибке конфигурации.",
        "criteria_weights": "{}"
    })
    
    vacancy_data_for_prompt = {
        "title": selected_vacancy_info["title"],
        "requirements": selected_vacancy_info.get("requirements", "Требования не указаны."),
        "questions_list": "\n".join(selected_vacancy_info["questions"]),
        "script": selected_vacancy_info["script"],
        "criteria_weights": selected_vacancy_info.get("criteria_weights", "{}")
    }

    # 3. Инициализация файла отчета
    report_path = REPORTS_DIR / f"{interview_id}.json"
    report_data = {
        "interview_id": interview_id,
        "vacancy_title": vacancy_data_for_prompt["title"],
        "date": datetime.utcnow().isoformat(),
        "dialogue": []
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Новое собеседование. ID: {interview_id}. Отчет создан: {report_path}")

    # 4. Приветствие
    initial_response = f"Здравствуйте! Давайте начнем собеседование на вакансию {selected_vacancy_info['title']}. Представьтесь, пожалуйста, и расскажите немного о себе."
    chat_history[interview_id].append(("assistant", initial_response))
    
    # Сохранение приветственного сообщения в отчет
    report_data["dialogue"].append({
        "assistant_text": initial_response,
        "user_text": None,
        "emotion": None,
        "structure_analysis": None,
        "internal_analysis": "Приветственное сообщение ассистента."
    })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    initial_chunks = split_text_into_chunks(initial_response)
    for chunk_text in initial_chunks:
        await websocket.send_json({"type": "text", "sender": "assistant", "data": chunk_text})
        stream = tts_model.inference_stream(chunk_text, "ru", gpt_cond_latent, speaker_embedding, speed=1.0)
        for chunk_audio in stream:
            await websocket.send_bytes(chunk_audio.squeeze().cpu().numpy().tobytes())
    await websocket.send_json({"type": "audio_end"})

    # 5. Основной цикл общения
    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "end_interview":
                    farewell_message = "Спасибо за уделенное время! Мы свяжемся с вами в ближайшее время! До свидания."
                    chat_history[interview_id].append(("assistant", farewell_message))
                    
                    farewell_chunks = split_text_into_chunks(farewell_message)
                    for chunk_text in farewell_chunks:
                        await websocket.send_json({"type": "text", "sender": "assistant", "data": chunk_text})
                        stream = tts_model.inference_stream(chunk_text, "ru", gpt_cond_latent, speaker_embedding, speed=1.0)
                        for chunk_audio in stream:
                            await websocket.send_bytes(chunk_audio.squeeze().cpu().numpy().tobytes())
                    
                    await websocket.send_json({"type": "audio_end"})
                    # Сохранение финального сообщения в отчет
                    report_data["dialogue"].append({
                        "assistant_text": farewell_message,
                        "user_text": "[Собеседование завершено]",
                        "emotion": None,
                        "structure_analysis": None,
                        "internal_analysis": "Собеседование завершено кандидатом."
                    })
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, ensure_ascii=False, indent=2)
                    await asyncio.sleep(0.1)
                    await websocket.close(code=1000)
                    break

            elif "bytes" in message:
                contents = message["bytes"]
                try:
                    # Декодирование аудио
                    audio_segment = AudioSegment.from_file(io.BytesIO(contents))
                    wav_buffer = io.BytesIO()
                    audio_segment.export(wav_buffer, format="wav")
                    wav_buffer.seek(0)
                    audio_float, sr = sf.read(wav_buffer, dtype='float32')
                    if sr != SAMPLE_RATE:
                        audio_float = resampy.resample(audio_float, sr, SAMPLE_RATE, filter='kaiser_fast')

                    # ASR: Преобразование речи в текст
                    segments, _ = asr_model.transcribe(audio_float, language=None, beam_size=5)
                    user_text = "".join([s.text for s in segments]).strip()
                    
                    full_assistant_response = ""
                    if not user_text:
                        await websocket.send_json({"type": "text", "sender": "user", "data": "[неразборчиво]"})
                        full_assistant_response = "Извините, я вас не расслышал. Не могли бы вы повторить?"
                        chat_history[interview_id].append(("assistant", full_assistant_response))
                        report_data["dialogue"].append({
                            "assistant_text": full_assistant_response,
                            "user_text": "[неразборчиво]",
                            "emotion": None,
                            "structure_analysis": None,
                            "pauses_info": "N/A",  # Дефолт
                            "internal_analysis": "Кандидат не предоставил разборчивый ответ."
                        })
                    else:
                        print(f"👤 Кандидат ({interview_id}): {user_text}")
                        await websocket.send_json({"type": "text", "sender": "user", "data": user_text})
                        
                        # Анализ речи: паузы, эмоции, структура
                        pauses_info = analyze_pauses(audio_float, SAMPLE_RATE)  # Новое
                        emotion = predict_emotion(audio_float)
                        structure_analysis = analyze_structure(user_text)
                        analysis_response = generate_and_analyze_response(
                            interview_id, user_text, emotion, pauses_info, structure_analysis, vacancy_data_for_prompt
                        )
                        full_assistant_response = analysis_response.assistant_response
                        
                        # Сохранение в отчёт с паузами
                        report_data["dialogue"].append({
                            "assistant_text": full_assistant_response,
                            "user_text": user_text,
                            "emotion": emotion,
                            "structure_analysis": structure_analysis,
                            "pauses_info": pauses_info,  # Новое поле
                            "internal_analysis": analysis_response.internal_analysis
                        })
                    
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"🤖 Ассистент ({interview_id}): {full_assistant_response}")
                    
                    # Отправка ответа по частям (стриминг)
                    response_chunks = split_text_into_chunks(full_assistant_response)
                    if not response_chunks:
                        await websocket.send_json({"type": "audio_end"})
                        continue

                    for chunk_text in response_chunks:
                        print(f"  -> Отправка чанка: '{chunk_text}'")
                        await websocket.send_json({"type": "text", "sender": "assistant", "data": chunk_text})
                        stream = tts_model.inference_stream(chunk_text, "ru", gpt_cond_latent, speaker_embedding, speed=1.0)
                        for chunk_audio in stream:
                            await websocket.send_bytes(chunk_audio.squeeze().cpu().numpy().tobytes())
                    
                    await websocket.send_json({"type": "audio_end"})

                except Exception as e:
                    print(f"Ошибка в цикле обработки аудио: {e}")
                    await websocket.send_json({"type": "audio_end"})  # Разблокируем кнопку на клиенте
                    continue

    except WebSocketDisconnect as e:
        print(f"Клиент отключился (ID: {interview_id}). Код: {e.code}")
        report_data["dialogue"].append({
            "assistant_text": None,
            "user_text": "[Клиент отключился]",
            "emotion": None,
            "structure_analysis": None,
            "internal_analysis": f"Собеседование прервано. Код отключения: {e.code}"
        })
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
    finally:
        print(f"Сессия {interview_id} завершена. Финальный отчет сохранен.")
        if interview_id in chat_history:
            del chat_history[interview_id]

# =======================
# Запуск сервера
# =======================
if __name__ == "__main__":
    print("✅ Сервер готов к запуску.")
    print("✅ Основное приложение доступно по адресу http://127.0.0.1:8000")
    print("✅ Админ-панель доступна по адресу http://127.0.0.1:8000/admin")
    print("✅ Архив отчетов доступен по адресу http://127.0.0.1:8000/archive")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)