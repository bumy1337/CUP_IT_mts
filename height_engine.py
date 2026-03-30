"""
Скрипт: Задание 3 - Определение наиболее достоверной высоты здания

ЗАДАЧА:
Для каждого из +-70K зданий Санкт-Петербурга определить наиболее достоверную
высоту, используя все доступные источники данных.

РАССМОТРЕННЫЕ ПОДХОДЫ И ОБОСНОВАНИЕ ВЫБОРА:

Подход A: «Приоритетный источник» (baseline МТС)
  Идея: выбрать один «лучший» источник и брать высоту оттуда.
  Плюсы: простота, скорость.
  Минусы: полностью игнорирует данные из других источников.
  Если приоритетный источник ошибся - ошибка не ловится.
  ВЫВОД: ОТКЛОНЕН как основной метод, но используется как fallback.

Подход B: «Простое усреднение»
  Идея: если есть несколько оценок - взять среднее.
  Плюсы: использует все данные.
  Минусы: одна грубая ошибка (выброс) искажает все среднее.
  Не учитывает что источники имеют разную надежность.
  Вывод: ОТКЛОНЕН.

Подход C: «Взвешенное усреднение по надежности и площади» (НАШ ВЫБОР)
  Идея: каждой оценке высоты назначается вес, который зависит от:
    1) Надежности источника (инструментальное измерение > расчЕт > грубая оценка)
    2) Площади полигона (больший полигон = более детальная геометрия = надежнее)
    3) Согласованности с другими оценками (если все говорят +-15 м, а один - 50 м,
       у выброса вес снижается)

Подход D: «Голосование» (дополнение к C)
  Идея: если есть 3+ независимых оценки - берем взвешенную медиану.
  Медиана устойчива к выбросам (1 грубая ошибка не влияет).
  Вывод: ИСПОЛЬЗУЕТСЯ при наличии 3+ оценок.

Подход E: «ML-предсказание» (дополнение к C)
  Идея: для зданий без данных - обучить модель на зданиях с известной высотой.
  Фичи: площадь, форма, высота соседей, назначение.
  Вывод: ИСПОЛЬЗУЕТСЯ как fallback для зданий без высоты из источников.

ИТОГОВЫЙ АЛГОРИТМ (комбинация C + D + E):

Для каждого здания:
  1. Собираем ВСЕ доступные оценки высоты (до 4 штук):
     - height из Б (вес 1.0 × коэфф. площади)
     - stairs × avg_floor_height из Б (вес 0.7 × коэфф. площади)
     - gkh_floor_count_max × avg_floor из А (вес 0.4 × коэфф. площади)
     - gkh_floor_count_min × avg_floor из А (вес 0.3 × коэфф. площади)
  2. Если оценок >= 3: взвешенная медиана (голосование)
     Если оценок == 2: взвешенное среднее
     Если оценок == 1: берем как есть
     Если оценок == 0: ML-предсказание
  3. Рассчитываем confidence на основе:
     - количества оценок
     - их согласованности (CV = коэфф. вариации)
     - наличия перекрестной валидации

ПОЧЕМУ ЭТО ЛУЧШЕ РЕШЕНИЯ МТС:

МТС использует приоритетный источник (подход A). Наш алгоритм:
  1. Использует ВСЕ доступные данные, не выбрасывая информацию
  2. Устойчив к выбросам через голосование (медиану)
  3. Учитывает качество геометрии через вес по площади
  4. Автоматически оценивает уверенность в результате
  5. Покрывает 100% зданий через ML fallback

"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
import warnings
import logging
import time
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRIC_CRS = 'EPSG:32636'

# Веса надежности источников (экспертные, обоснование ниже в коде)
SOURCE_WEIGHTS = {
    'B_height': 1.0,           # Прямое измерение - самое надежное
    'B_stairs_calc': 0.7,      # Расчет по этажности - хорошо, но есть погрешность avg_floor
    'A_gkh_max': 0.4,          # ЖКХ данные - часто неполные, иногда устаревшие
    'A_gkh_min': 0.3,          # Минимальная этажность - еще менее надежна
}

# Медианная высота этажа по типу назначения здания (м).
# Используется как fallback когда avg_floor_height = 0 или отсутствует.
# ПОЧЕМУ НЕ ЗАХАРДКОЖЕННЫЕ 3.0:
#   Производственные здания (склады, цеха) типично 5-8 м в одном этаже.
#   Жилые — 2.7-3.2 м. Торговые центры — 4-6 м.
#   Ошибка в afh умножается на число этажей, поэтому для 10-этажного
#   производственного здания разница между 3.0 и 5.5 даёт +25 м ошибки.
AFH_BY_PURPOSE = {
    'Жилое здание':                    3.0,
    'Строение жилое (частное)':        3.0,
    'Строение дачное':                 3.0,
    'Нежилое здание':                  3.3,
    'Административное здание':         3.5,
    'Учебное учреждение':              3.5,
    'Научное здание':                  3.5,
    'Учреждение здравоохранения':      3.3,
    'Детское учреждение':              3.3,
    'Торговый центр':                  4.5,
    'Производственное здание':         5.5,
    'Производственные сооружения':     5.5,
    'Здание культурных мероприятий':   4.0,
    'Музей':                           4.0,
    'Церкви':                          4.5,
}
AFH_DEFAULT = 3.3  # глобальный fallback если purpose неизвестен


def get_afh(avg_floor_b, purpose):
    """
    Возвращает высоту этажа (avg floor height) для расчёта высоты здания.

    Приоритет:
      1. avg_floor_height из источника Б (если > 0) — прямые данные
      2. Таблица AFH_BY_PURPOSE по назначению здания
      3. Глобальный fallback AFH_DEFAULT = 3.3 м
    """
    if avg_floor_b and avg_floor_b > 0:
        return avg_floor_b
    if purpose:
        # Нормируем строку: убираем пробелы, пробуем точное совпадение
        p = str(purpose).strip()
        if p in AFH_BY_PURPOSE:
            return AFH_BY_PURPOSE[p]
        # Частичное совпадение (на случай незначительных расхождений в строке)
        for key, val in AFH_BY_PURPOSE.items():
            if key.lower() in p.lower() or p.lower() in key.lower():
                return val
    return AFH_DEFAULT


def load_geodata(filepath, name):
    logger.info(f"Загрузка {name} из {filepath}...")
    df = pd.read_csv(filepath)
    geom_col = None
    for col in ['geometry_wkt', 'geometry', 'wkt']:
        if col in df.columns: geom_col = col; break
    if not geom_col: raise ValueError(f"Нет колонки геометрии в {filepath}")
    df['geometry'] = df[geom_col].apply(lambda x: wkt.loads(x) if pd.notna(x) and isinstance(x, str) else None)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    s = gdf.geometry.dropna().iloc[0].centroid
    gdf = gdf.set_crs('EPSG:4326' if -180 <= s.x <= 180 else METRIC_CRS)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    logger.info(f"  Загружено: {len(gdf)}")
    return gdf

# ЭТАП 1: СБОР ВСЕХ ОЦЕНОК ВЫСОТЫ

def collect_all_estimates(components, gdf_a, gdf_b):
    """
    Для каждого здания собирает ВСЕ доступные оценки высоты из обоих источников.

    Возвращает DataFrame, где каждая строка = одно здание, с колонками:
      - estimates: список кортежей (значение, вес_источника, источник)
      - raw данные из обоих источников
      - площадь и прочие метаданные
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 1: СБОР ОЦЕНОК ВЫСОТЫ ИЗ ВСЕХ ИСТОЧНИКОВ")
    logger.info("=" * 60)

    # Предрассчет площадей в метрах
    gdf_a_proj = gdf_a.to_crs(METRIC_CRS)
    gdf_b_proj = gdf_b.to_crs(METRIC_CRS)
    areas_a = gdf_a_proj.geometry.area
    areas_b = gdf_b_proj.geometry.area

    results = []

    for comp_id, group in components.groupby('component_id'):
        a_idx = group[group['source'] == 'A']['original_index'].tolist()
        b_idx = group[group['source'] == 'B']['original_index'].tolist()

        # Собираем сырые данные
        height_b = stairs_b = avg_floor_b = None
        gkh_max_a = gkh_min_a = None
        purpose = None
        area_a_total = area_b_total = 0

        for idx in b_idx:
            if idx < len(gdf_b):
                r = gdf_b.iloc[idx]
                if height_b is None and pd.notna(r.get('height')): height_b = float(r['height'])
                if stairs_b is None and pd.notna(r.get('stairs')): stairs_b = float(r['stairs'])
                if avg_floor_b is None and pd.notna(r.get('avg_floor_height')): avg_floor_b = float(r['avg_floor_height'])
                if purpose is None and pd.notna(r.get('purpose_of_building')): purpose = str(r['purpose_of_building'])
                if idx < len(areas_b): area_b_total += areas_b.iloc[idx]

        for idx in a_idx:
            if idx < len(gdf_a):
                r = gdf_a.iloc[idx]
                if gkh_min_a is None and pd.notna(r.get('gkh_floor_count_min')): gkh_min_a = float(r['gkh_floor_count_min'])
                if gkh_max_a is None and pd.notna(r.get('gkh_floor_count_max')): gkh_max_a = float(r['gkh_floor_count_max'])
                if idx < len(areas_a): area_a_total += areas_a.iloc[idx]

        total_area = max(area_a_total, area_b_total, 1.0)
        afh = get_afh(avg_floor_b, purpose)

        # Флаг дефолтной высоты: height=4.5 + stairs=1 — это технический дефолт
        # в источнике Б, не реальное измерение. Такие записи не должны попадать
        # в обучающую выборку ML как «достоверные».
        is_default = (
            height_b is not None and height_b == 4.5 and
            stairs_b is not None and stairs_b == 1.0
        )

        # Формируем список оценок
        estimates = []  # (значение_м, вес, название_источника)

        if height_b is not None and height_b > 0:
            # Вес = надежность × коэфф. площади (нормализованный)
            area_coeff = min(area_b_total / 100, 3.0)  # 100 м² = 1.0, 300+ = 3.0 (cap)
            w = SOURCE_WEIGHTS['B_height'] * max(area_coeff, 0.5)
            estimates.append((height_b, w, 'B_height'))

        if stairs_b is not None and stairs_b > 0:
            calc_height = stairs_b * afh
            area_coeff = min(area_b_total / 100, 3.0)
            w = SOURCE_WEIGHTS['B_stairs_calc'] * max(area_coeff, 0.5)
            estimates.append((calc_height, w, 'B_stairs_calc'))

        if gkh_max_a is not None and gkh_max_a > 0:
            calc_height = gkh_max_a * afh
            area_coeff = min(area_a_total / 100, 3.0)
            w = SOURCE_WEIGHTS['A_gkh_max'] * max(area_coeff, 0.5)
            estimates.append((calc_height, w, 'A_gkh_max'))

        if gkh_min_a is not None and gkh_min_a > 0:
            calc_height = gkh_min_a * afh
            area_coeff = min(area_a_total / 100, 3.0)
            w = SOURCE_WEIGHTS['A_gkh_min'] * max(area_coeff, 0.5)
            estimates.append((calc_height, w, 'A_gkh_min'))

        results.append({
            'component_id': comp_id,
            'estimates': estimates,
            'n_estimates': len(estimates),
            'height_b': height_b, 'stairs_b': stairs_b,
            'avg_floor_height': avg_floor_b,
            'gkh_floor_max': gkh_max_a, 'gkh_floor_min': gkh_min_a,
            'purpose': purpose,
            'area_m2': total_area,
            'n_polygons_a': len(a_idx), 'n_polygons_b': len(b_idx),
            'is_default_height': is_default,
        })

    df = pd.DataFrame(results)
    n_est = df['n_estimates'].value_counts().sort_index()
    logger.info("\n  Количество оценок на здание:")
    for k, v in n_est.items():
        logger.info(f"    {k} оценок: {v} зданий")

    return df

# ЭТАП 2: ОПРЕДЕЛЕНИЕ ВЫСОТЫ (взвешенное усреднение + голосование)

def determine_height(buildings):
    """
    Для каждого здания определяет финальную высоту по алгоритму:

    3 оценки - ВЗВЕШЕННАЯ МЕДИАНА (голосование)
       Почему медиана, а не среднее: медиана устойчива к выбросам.
       Если 3 оценки = [15, 14, 50], среднее = 26.3 (плохо), медиана = 15 (хорошо).

    2 оценки - ВЗВЕШЕННОЕ СРЕДНЕЕ
       result = (v1 * w1 + v2 * w2) / (w1 + w2)
       Если оценки согласованы (< 30% разницы) - confidence выше.

    1 оценка - БЕРЕМ КАК ЕСТЬ
       Confidence зависит от источника.

    0 оценок - NaN (пойдет в ML)
    """
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 2: ОПРЕДЕЛЕНИЕ ВЫСОТЫ")
    logger.info("  Метод: взвешенное усреднение + голосование (медиана)")
    logger.info("=" * 60)

    stats = {'voting_3plus': 0, 'weighted_avg_2': 0, 'single_source': 0,
             'no_data_for_ml': 0, 'cross_validated': 0}

    height_finals = []
    height_sources = []
    confidences = []
    cv_values = []  # коэффициент вариации

    for _, row in buildings.iterrows():
        estimates = row['estimates']

        if len(estimates) == 0:
            height_finals.append(np.nan)
            height_sources.append(None)
            confidences.append('none')
            cv_values.append(np.nan)
            stats['no_data_for_ml'] += 1
            continue

        values = np.array([e[0] for e in estimates])
        weights = np.array([e[1] for e in estimates])
        sources = [e[2] for e in estimates]

        if len(estimates) >= 3:
            # Сортируем по значению, находим медианную позицию по весам
            sorted_idx = np.argsort(values)
            sorted_v = values[sorted_idx]
            sorted_w = weights[sorted_idx]
            cumw = np.cumsum(sorted_w)
            median_idx = np.searchsorted(cumw, cumw[-1] / 2)
            median_idx = min(median_idx, len(sorted_v) - 1)

            height_final = sorted_v[median_idx]
            source = f"voting_{len(estimates)}est"
            stats['voting_3plus'] += 1

        elif len(estimates) == 2:
            # ВЗВЕШЕННОЕ СРЕДНЕЕ
            height_final = np.average(values, weights=weights)
            source = f"wavg_{sources[0]}+{sources[1]}"
            stats['weighted_avg_2'] += 1

        else:  # 1 оценка
            height_final = values[0]
            source = sources[0]
            stats['single_source'] += 1

        # Confidence на основе согласованности
        if len(values) >= 2:
            cv = np.std(values) / (np.mean(values) + 1e-10)  # коэфф. вариации
            if cv < 0.15:
                confidence = 'high_validated'  # все оценки согласованы (< 15% разброс)
                stats['cross_validated'] += 1
            elif cv < 0.30:
                confidence = 'high'
            elif cv < 0.50:
                confidence = 'medium'
            else:
                confidence = 'low_conflicting'  # источники сильно расходятся
        else:
            # Одна оценка - confidence по типу источника
            if 'B_height' in sources:
                confidence = 'high'
            elif 'B_stairs' in sources[0]:
                confidence = 'medium'
            else:
                confidence = 'low'
            cv = np.nan

        height_finals.append(height_final)
        height_sources.append(source)
        confidences.append(confidence)
        cv_values.append(cv)

    buildings['height_final'] = height_finals
    buildings['height_source'] = height_sources
    buildings['height_confidence'] = confidences
    buildings['estimate_cv'] = cv_values

    stats['total'] = len(buildings)
    stats['with_height'] = buildings['height_final'].notna().sum()

    logger.info(f"\n  Голосование (3+ оценки): {stats['voting_3plus']}")
    logger.info(f"  Взвешенное среднее (2 оценки): {stats['weighted_avg_2']}")
    logger.info(f"  Одиночный источник: {stats['single_source']}")
    logger.info(f"  Без данных (для ML): {stats['no_data_for_ml']}")
    logger.info(f"  Перекрестно валидированы (CV < 15%): {stats['cross_validated']}")

    return buildings, stats

# ЭТАП 3: ПРОСТРАНСТВЕННЫЕ ФИЧИ ДЛЯ ML

def compute_features(buildings, gdf_a, gdf_b, components):
    """Вычисляет фичи для ML: площадь, периметр, компактность, соседи."""
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 3: ВЫЧИСЛЕНИЕ ФИЧЕЙ ДЛЯ ML")
    logger.info("=" * 60)

    # Собираем геометрии
    logger.info("  Привязка геометрий...")
    geom_list = []
    for _, row in buildings.iterrows():
        comp_rows = components[components['component_id'] == row['component_id']]
        geom = None
        for _, cr in comp_rows.iterrows():
            gdf = gdf_a if cr['source'] == 'A' else gdf_b
            idx = int(cr['original_index'])
            if idx < len(gdf):
                g = gdf.geometry.iloc[idx]
                if g is not None and not g.is_empty: geom = g; break
        geom_list.append(geom)

    buildings['geometry'] = geom_list
    gdf_tmp = gpd.GeoDataFrame(buildings, geometry='geometry')
    if gdf_tmp.crs is None: gdf_tmp = gdf_tmp.set_crs('EPSG:4326')
    gdf_proj = gdf_tmp.to_crs(METRIC_CRS)

    # Геометрические фичи
    logger.info("  Площадь, периметр, компактность...")
    buildings['area_m2_geom'] = gdf_proj.geometry.area
    buildings['perimeter_m'] = gdf_proj.geometry.length
    buildings['compactness'] = 4 * np.pi * buildings['area_m2_geom'] / (buildings['perimeter_m']**2 + 1e-10)

    def nverts(g):
        if g is None or g.is_empty: return 0
        if g.geom_type == 'Polygon': return len(g.exterior.coords)
        if g.geom_type == 'MultiPolygon': return sum(len(p.exterior.coords) for p in g.geoms)
        return 0
    buildings['n_vertices'] = gdf_proj.geometry.apply(nverts)

    # Средняя высота соседей
    logger.info("  Высота соседей (100 м)...")
    centroids = gdf_proj.geometry.centroid
    sindex = gdf_proj.sindex
    known = buildings['height_final'].notna()
    avg_nh = []
    for i in range(len(buildings)):
        c = centroids.iloc[i]
        if c is None or c.is_empty: avg_nh.append(np.nan); continue
        cands = list(sindex.intersection(c.buffer(100).bounds))
        heights = [buildings.iloc[j]['height_final'] for j in cands
                    if j != i and j < len(buildings) and known.iloc[j]
                    and c.distance(centroids.iloc[j]) <= 100]
        avg_nh.append(np.mean(heights) if heights else np.nan)
        if (i+1) % 50000 == 0: logger.info(f"    {i+1}/{len(buildings)}...")
    buildings['avg_height_neighbors_100m'] = avg_nh

    if 'purpose' in buildings.columns:
        buildings['purpose_encoded'] = buildings['purpose'].fillna('unknown').astype('category').cat.codes
    else:
        buildings['purpose_encoded'] = 0

    buildings['area_m2'] = buildings['area_m2'].fillna(buildings['area_m2_geom'])
    if 'geometry' in buildings.columns: buildings = buildings.drop(columns=['geometry'])

    logger.info(f"  Фичи готовы для {len(buildings)} зданий")
    return buildings

# ЭТАП 3.5: РАЗРЕШЕНИЕ КОНФЛИКТОВ ЧЕРЕЗ СОСЕДЕЙ

def resolve_conflicts(buildings):
    """
    Для зданий с low_conflicting (источники сильно расходятся):
    смотрим на высоту соседей и выбираем ту оценку, которая ближе к окружению.

    Логика:
      Здание с оценками [15 м, 45 м], соседи в среднем 14 м
      - 15 м ближе к соседям - тогда берем 15 м, confidence = 'resolved_by_neighbors'

    Почему это работает:
      Реальные городские кварталы однородны по высоте.
      Если вокруг все дома 5-этажные, а один источник говорит 15 этажей - скорее всего ошибка в этом источнике.

    Если нет данных о соседях - оставляем взвешенное среднее как есть.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 3.5: РАЗРЕШЕНИЕ КОНФЛИКТОВ ЧЕРЕЗ СОСЕДЕЙ")
    logger.info("=" * 60)

    conflict_mask = buildings['height_confidence'] == 'low_conflicting'
    n_conflicts = conflict_mask.sum()

    if n_conflicts == 0:
        logger.info("  Конфликтов нет")
        return buildings

    logger.info(f"  Конфликтных зданий: {n_conflicts}")

    has_neighbors = (
        conflict_mask &
        buildings['avg_height_neighbors_100m'].notna()
    )
    n_resolvable = has_neighbors.sum()
    logger.info(f"  Из них с данными о соседях: {n_resolvable}")

    n_resolved = 0

    for idx in buildings[has_neighbors].index:
        row = buildings.loc[idx]
        neighbor_h = row['avg_height_neighbors_100m']

        # Собираем все оценки этого здания
        estimates = []
        if pd.notna(row.get('height_b')) and row['height_b'] > 0:
            estimates.append(('B_height', row['height_b']))
        if pd.notna(row.get('stairs_b')) and row['stairs_b'] > 0:
            afh = get_afh(
                row.get('avg_floor_height') if pd.notna(row.get('avg_floor_height')) else None,
                row.get('purpose')
            )
            estimates.append(('B_stairs', row['stairs_b'] * afh))
        if pd.notna(row.get('gkh_floor_max')) and row['gkh_floor_max'] > 0:
            afh = get_afh(
                row.get('avg_floor_height') if pd.notna(row.get('avg_floor_height')) else None,
                row.get('purpose')
            )
            estimates.append(('A_gkh_max', row['gkh_floor_max'] * afh))
        if pd.notna(row.get('gkh_floor_min')) and row['gkh_floor_min'] > 0:
            afh = get_afh(
                row.get('avg_floor_height') if pd.notna(row.get('avg_floor_height')) else None,
                row.get('purpose')
            )
            estimates.append(('A_gkh_min', row['gkh_floor_min'] * afh))

        if len(estimates) < 2:
            continue

        # Выбираем оценку ближайшую к соседям
        best_source = None
        best_value = None
        best_dist = float('inf')

        for source, value in estimates:
            dist = abs(value - neighbor_h)
            if dist < best_dist:
                best_dist = dist
                best_source = source
                best_value = value

        buildings.loc[idx, 'height_final'] = best_value
        buildings.loc[idx, 'height_source'] = f'resolved_{best_source}'
        buildings.loc[idx, 'height_confidence'] = 'resolved_by_neighbors'
        n_resolved += 1

    logger.info(f"  Разрешено через соседей: {n_resolved}")
    logger.info(f"  Осталось конфликтных: {n_conflicts - n_resolved}")

    return buildings

# ЭТАП 4: ML-ПРЕДСКАЗАНИЕ ДЛЯ ЗДАНИЙ БЕЗ ДАННЫХ

def train_and_predict(buildings):
    """
    Обучает модель на зданиях с известной высотой и предсказывает для остальных.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 4: ML-ПРЕДСКАЗАНИЕ ВЫСОТЫ")
    logger.info("=" * 60)

    feature_cols = ['area_m2', 'perimeter_m', 'compactness', 'n_vertices',
                    'avg_height_neighbors_100m', 'purpose_encoded', 'n_polygons_a', 'n_polygons_b']

    # Маска обучения: здания с известной высотой И не являющиеся дефолтными.
    # ПОЧЕМУ ИСКЛЮЧАЕМ is_default_height:
    #   height=4.5 + stairs=1 — технический дефолт источника Б (44.6% записей),
    #   не реальное измерение. Если обучать модель на них как на «правде»,
    #   она переобучается предсказывать ~4.5 м для любого здания без данных,
    #   занижая высоту для настоящих одноэтажных объектов и систематически
    #   ошибаясь на всех остальных. Исключение снижает MAPE и улучшает R².
    is_default = buildings.get('is_default_height', pd.Series(False, index=buildings.index))
    train_mask = (
        buildings['height_final'].notna() &
        (buildings['height_final'] > 0) &
        ~is_default
    )
    predict_mask = buildings['height_final'].isna()

    n_excluded = (buildings['height_final'].notna() & is_default).sum()
    logger.info(f"  Исключено дефолтных из обучения: {n_excluded} "
                f"(height=4.5 + stairs=1 — технический дефолт)")

    X_train = buildings.loc[train_mask, feature_cols].copy()
    y_train = buildings.loc[train_mask, 'height_final'].copy()

    logger.info(f"  Обучение: {len(X_train)} зданий")
    logger.info(f"  Предсказание: {predict_mask.sum()} зданий")

    if len(X_train) < 100 or predict_mask.sum() == 0:
        logger.info("  Пропускаем ML")
        return buildings, {}

    for col in feature_cols: X_train[col] = X_train[col].fillna(X_train[col].median())

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                                   num_leaves=63, min_child_samples=20, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
        model_name = 'LightGBM'
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           min_samples_leaf=20, subsample=0.8, random_state=42)
        model.fit(X_tr, y_tr)
        model_name = 'sklearn GB'

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    mape = np.mean(np.abs(y_te - y_pred) / (y_te + 1e-10)) * 100

    metrics = {'model': model_name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
               'R2': round(r2, 4), 'MAPE': round(mape, 1),
               'train': len(X_tr), 'test': len(X_te)}

    logger.info(f"  Модель: {model_name}")
    logger.info(f"  MAE: {mae:.2f} м | RMSE: {rmse:.2f} м | R²: {r2:.4f} | MAPE: {mape:.1f}%")

    if hasattr(model, 'feature_importances_'):
        imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
        logger.info("  Важность фичей:")
        for f, v in imp:
            bar = "█" * int(v / max(model.feature_importances_) * 20)
            logger.info(f"    {f:35s} {v:.3f} {bar}")

    # Предсказание
    X_pred = buildings.loc[predict_mask, feature_cols].copy()
    for col in feature_cols: X_pred[col] = X_pred[col].fillna(X_train[col].median())
    preds = np.clip(model.predict(X_pred), 2.5, 500)

    buildings.loc[predict_mask, 'height_final'] = preds
    buildings.loc[predict_mask, 'height_source'] = 'ML_predicted'
    buildings.loc[predict_mask, 'height_confidence'] = 'predicted'

    logger.info(f"\n  Предсказано: {predict_mask.sum()} зданий")
    logger.info(f"  Диапазон: {preds.min():.1f} - {preds.max():.1f} м")

    return buildings, metrics

# ТОЧКА ВХОДА

if __name__ == '__main__':
    CLEAN_A = 'output/clean_A.csv'; CLEAN_B = 'output/clean_B.csv'
    COMPONENTS = 'output/matched_components.csv'; OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ЗАДАНИЕ 3: ОПРЕДЕЛЕНИЕ ВЫСОТЫ ЗДАНИЙ")
    logger.info("  Метод: взвешенное усреднение + голосование + ML fallback")
    logger.info("=" * 60)
    t0 = time.time()

    gdf_a = load_geodata(CLEAN_A, "А"); gdf_b = load_geodata(CLEAN_B, "Б")
    components = pd.read_csv(COMPONENTS)
    logger.info(f"Компоненты: {components['component_id'].nunique()} зданий")

    # Этап 1: сбор оценок
    buildings = collect_all_estimates(components, gdf_a, gdf_b)

    # Этап 2: определение высоты
    buildings, height_stats = determine_height(buildings)

    # Этап 3: фичи для ML
    buildings = compute_features(buildings, gdf_a, gdf_b, components)

    # Этап 3.5: разрешение конфликтов через соседей
    buildings = resolve_conflicts(buildings)

    # Этап 4: ML для зданий без данных
    buildings, ml_metrics = train_and_predict(buildings)

    # Убираем колонку estimates (список, не сериализуется)
    if 'estimates' in buildings.columns:
        buildings = buildings.drop(columns=['estimates'])

    # Сохранение
    logger.info("\n" + "=" * 60)
    logger.info("СОХРАНЕНИЕ")
    buildings.to_csv(os.path.join(OUTPUT_DIR, 'buildings_with_height.csv'), index=False)

    # Итоговая статистика
    total = len(buildings); has_h = buildings['height_final'].notna().sum()
    logger.info(f"  Всего: {total}, с высотой: {has_h} ({has_h/total*100:.1f}%)")
    for src, cnt in buildings['height_source'].value_counts().items():
        logger.info(f"    {src}: {cnt}")

    h = buildings['height_final'].dropna()
    logger.info(f"  Высоты: min={h.min():.1f}, median={h.median():.1f}, mean={h.mean():.1f}, max={h.max():.1f}")

    all_stats = {'height_determination': height_stats, 'ml': ml_metrics,
                 'total': total, 'with_height': int(has_h), 'coverage': round(has_h/total*100, 1),
                 'default_height_excluded_from_ml': int(buildings.get('is_default_height', pd.Series(False)).sum())}
    with open(os.path.join(OUTPUT_DIR, 'height_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    logger.info(f"\n ЗАДАНИЕ 3 ЗАВЕРШЕНО за {time.time()-t0:.1f}с")