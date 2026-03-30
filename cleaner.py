"""

Скрипт 1: Очистка данных (Задание 1)

Обрабатывает типы невалидной геометрии 1, 4, 5, 6, 7:
  1. Самопересечение - make_valid()
  4. Нулевая площадь - удаление (area == 0)
  5. Неправильная ориентация - shapely.normalize()
  6. Вырожденные отверстия - buffer(0)
  7. Пустая геометрия - удаление (None / empty)

Улучшения по сравнению с базовым решением МТС:
  - make_valid + buffer(0) применяется КО ВСЕМ геометриям (не только к is_valid==False)
  - Нормализация ориентации колец (normalize)
  - Удаление точных дубликатов геометрий (побайтовое сравнение WKB)
  - Двухступенчатая фильтрация выбросов: жесткие пороги + GMM (бимодальное распределение)
  - Проверка согласованности height vs stairs * avg_floor_height
  - Детальная диагностика на каждом шаге (до/после)
  - Площадь считается в метрической проекции (UTM 36N, EPSG:32636), не в градусах
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import warnings
import logging
import time
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Метрическая проекция для Санкт-Петербурга
# UTM Zone 36N - стандартная проекция, покрывает 30°-36°E (СПб на 30.3°E)

METRIC_CRS = 'EPSG:32636'

# ЗАГРУЗКА ДАННЫХ

def _smart_read_csv(filepath: str) -> pd.DataFrame:
    """Умное чтение CSV: пробует разные разделители и кодировки."""
    for encoding in ['utf-8', 'cp1251', 'latin-1']:
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=encoding, nrows=5)
                if len(df.columns) > 2:
                    logger.info(f"  Формат: sep='{sep}', encoding='{encoding}'")
                    return pd.read_csv(filepath, sep=sep, encoding=encoding)
            except Exception:
                continue
    return pd.read_csv(filepath, sep=None, engine='python')


def _find_geometry_column(df: pd.DataFrame) -> str:
    """Автоопределение колонки с WKT-геометрией."""
    for name in ['geometry', 'wkt', 'geom', 'WKT', 'GEOMETRY']:
        if name in df.columns:
            sample = str(df[name].dropna().iloc[0]) if df[name].notna().any() else ''
            if 'POLYGON' in sample.upper() or 'POINT' in sample.upper():
                return name
    for col in df.columns:
        try:
            sample = str(df[col].dropna().iloc[0])
            if 'POLYGON' in sample.upper():
                return col
        except Exception:
            continue
    raise ValueError(f"Не найдена колонка с WKT! Колонки: {list(df.columns)}")


def _parse_wkt_safe(wkt_string):
    """Безопасный парсинг WKT."""
    if pd.isna(wkt_string) or not isinstance(wkt_string, str):
        return None
    try:
        return wkt.loads(wkt_string)
    except Exception:
        return None


def _detect_crs(gdf: gpd.GeoDataFrame) -> str:
    """Определяет CRS по координатам."""
    sample_geom = gdf.geometry.dropna().iloc[0]
    centroid = sample_geom.centroid
    if -180 <= centroid.x <= 180 and -90 <= centroid.y <= 90:
        return 'EPSG:4326'
    return METRIC_CRS


def load_source(filepath: str, name: str) -> gpd.GeoDataFrame:
    """Универсальная загрузка источника."""
    logger.info(f"Загрузка {name} из {filepath}...")
    df = _smart_read_csv(filepath)

    geom_col = _find_geometry_column(df)
    logger.info(f"  Колонка геометрии: '{geom_col}'")

    df['geometry'] = df[geom_col].apply(_parse_wkt_safe)
    n_failed = df['geometry'].isna().sum() - df[geom_col].isna().sum()
    if n_failed > 0:
        logger.warning(f"  Не удалось распарсить WKT: {n_failed} записей")

    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    crs = _detect_crs(gdf)
    gdf = gdf.set_crs(crs)

    logger.info(f"  CRS: {crs}")
    logger.info(f"  Загружено: {len(gdf)} записей")
    logger.info(f"  Колонки: {list(gdf.columns)}")
    logger.info(f"  Пропуски геометрии: {gdf.geometry.isna().sum()}")

    return gdf

# ОЧИСТКА ГЕОМЕТРИИ (все 7 типов)

def extract_polygons(geom):
    """
    Извлекает полигональные компоненты из любой геометрии.
    make_valid() может вернуть GeometryCollection с линиями/точками -
    нужны только Polygon и MultiPolygon.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom
    if geom.geom_type == 'MultiPolygon':
        return geom
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms
                 if g.geom_type in ('Polygon', 'MultiPolygon') and not g.is_empty and g.area > 0]
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        # Объединяем все полигоны в MultiPolygon
        return unary_union(polys)
    return None


def fix_all_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Комплексное исправление геометрии - обрабатывает типы 1, 4, 5, 6, 7.
    Стратегия (применяется КО ВСЕМ записям, а не только к невалидным):
      1. Удаляем пустые геометрии (тип 7)
      2. normalize() - исправляет ориентацию колец (тип 5)
      3. make_valid() - исправляет самопересечения (тип 1)
      4. buffer(0) - пересчитывает топологию, чинит вырожденные отверстия (тип 6)
      5. Извлекаем только полигоны из GeometryCollection
      6. Удаляем объекты с нулевой площадью (тип 4)
    ПОЧЕМУ КО ВСЕМ: Shapely is_valid не ловит все проблемы.
    Например, неправильная ориентация колец не считается невалидной,
    но может давать отрицательную площадь или ошибки при пересечении.
    """
    stats = {}
    initial = len(gdf)
    stats['initial'] = initial

    #  Шаг 1: Удаляем пустые (тип 7)
    mask_empty = gdf.geometry.isna() | gdf.geometry.is_empty
    n_empty = mask_empty.sum()
    gdf = gdf[~mask_empty].copy()
    stats['removed_empty'] = n_empty
    logger.info(f"  [Тип 7] Удалено пустых геометрий: {n_empty}")

    #  Шаг 2: Считаем невалидных ДО исправления
    n_invalid_before = (~gdf.geometry.is_valid).sum()
    stats['invalid_before'] = n_invalid_before
    logger.info(f"  Невалидных геометрий до исправления: {n_invalid_before} "
                f"({n_invalid_before/len(gdf)*100:.2f}%)")

    #  Шаг 3: normalize - make_valid - buffer(0) КО ВСЕМ
    logger.info(f"  Применяем normalize + make_valid + buffer(0) ко всем {len(gdf)} записям...")

    def repair_geometry(geom):
        """Полный цикл исправления одной геометрии."""
        if geom is None or geom.is_empty:
            return None
        try:
            # Шаг 5 (ориентация): normalize
            geom = geom.normalize()

            # Шаг 1 (самопересечения): make_valid
            if not geom.is_valid:
                geom = make_valid(geom)

            # Шаг 6 (вырожденные отверстия): buffer(0)
            geom = geom.buffer(0)

            # Извлекаем только полигоны
            geom = extract_polygons(geom)

            return geom
        except Exception:
            return None

    gdf['geometry'] = gdf['geometry'].apply(repair_geometry)

    #  Шаг 4: Удаляем то, что не удалось исправить
    mask_failed = gdf.geometry.isna() | gdf.geometry.is_empty
    n_failed = mask_failed.sum()
    gdf = gdf[~mask_failed].copy()
    stats['repair_failed'] = n_failed
    if n_failed > 0:
        logger.info(f"  Не удалось исправить (удалено): {n_failed}")

    #  Шаг 5: Проверка после исправления
    n_invalid_after = (~gdf.geometry.is_valid).sum()
    stats['invalid_after'] = n_invalid_after
    logger.info(f"  Невалидных после исправления: {n_invalid_after}")

    #  Шаг 6: Удаляем нулевую площадь (тип 4)
    # Считаем площадь в метрической проекции
    gdf_proj = gdf.to_crs(METRIC_CRS)
    area_m2 = gdf_proj.geometry.area
    mask_zero = area_m2 <= 0
    n_zero = mask_zero.sum()
    gdf = gdf[~mask_zero].copy()
    stats['removed_zero_area'] = n_zero
    if n_zero > 0:
        logger.info(f"  [Тип 4] Удалено с нулевой площадью: {n_zero}")

    stats['after_geometry_fix'] = len(gdf)
    logger.info(f"  После исправления геометрии: {initial} - {len(gdf)}")

    return gdf, stats

# УДАЛЕНИЕ ДУБЛИКАТОВ

def remove_exact_duplicates(gdf: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
    """
    Удаление точных дубликатов геометрий.
    Два полигона считаются дубликатами если их WKT идентичны.
    """
    before = len(gdf)
    gdf['_wkt_hash'] = gdf.geometry.apply(lambda g: g.wkb_hex if g is not None else None)
    gdf = gdf.drop_duplicates(subset='_wkt_hash', keep='first').copy()
    gdf = gdf.drop(columns=['_wkt_hash'])
    removed = before - len(gdf)
    if removed > 0:
        logger.info(f"  Удалено точных дубликатов: {removed}")
    return gdf

# ФИЛЬТРАЦИЯ ВЫБРОСОВ ПО АТРИБУТАМ

def gmm_outlier_filter(series: pd.Series, n_components: int = 2,
                       threshold_sigma: float = 3.0, col_name: str = '') -> tuple:
    """
    Двухступенчатая фильтрация выбросов с учетом бимодального распределения.

    ПОЧЕМУ НЕ IQR:
    Этажность в городе - бимодальное распределение (два горба):
      Горб 1: малоэтажки (2-5 этажей) - исторический центр, хрущевки
      Горб 2: высотки (9-25 этажей) - панельки, новостройки
    IQR считает медиану +-= 5, Q3 +-= 9, и все выше 16 этажей - «аномалия».
    Но это нормальные дома.

    РЕШЕНИЕ - GMM (Gaussian Mixture Model):
    Подбирает N нормальных распределений одновременно.
    Каждому значению назначает вероятность принадлежности к одному из кластеров.
    Выброс = значение, которое не попадает ни в один кластер
    (log-likelihood ниже порога).

    Ступень 1: жесткие пороги (вызывающий код убирает явный мусор ДО вызова GMM)
    Ступень 2: GMM на очищенных данных

    Args:
        series: данные (с NaN)
        n_components: сколько «горбов» искать (2 - бимодальное)
        threshold_sigma: сколько сигм от центра кластера = выброс
        col_name: название колонки (для логирования)

    Returns:
        (очищенная series, кол-во выбросов, dict с инфо о кластерах)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning(f"    sklearn не установлен! pip install scikit-learn")
        logger.warning(f"    GMM {col_name}: пропускаем, используем только жесткие пороги")
        return series, 0, {}

    valid_mask = series.notna()
    valid = series[valid_mask].values.reshape(-1, 1)

    if len(valid) < 100:
        logger.info(f"    GMM {col_name}: слишком мало данных ({len(valid)}), пропускаем")
        return series, 0, {}

    # Подбираем GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
    gmm.fit(valid)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # Логируем найденные кластеры
    cluster_info = {}
    for k in range(n_components):
        cluster_info[f'cluster_{k}'] = {
            'mean': float(means[k]),
            'std': float(stds[k]),
            'weight': float(weights[k]),
            'range': f'[{means[k] - threshold_sigma * stds[k]:.1f}, '
                     f'{means[k] + threshold_sigma * stds[k]:.1f}]'
        }
        logger.info(f"    GMM {col_name} кластер {k}: "
                    f"μ={means[k]:.1f}, σ={stds[k]:.1f}, "
                    f"вес={weights[k]:.2f}, "
                    f"допуск={cluster_info[f'cluster_{k}']['range']}")

    # Выброс = не попадает ни в один кластер в пределах threshold_sigma * sigma
    outlier_mask_values = np.ones(len(valid), dtype=bool)  # изначально все - выбросы
    for k in range(n_components):
        in_cluster = np.abs(valid.flatten() - means[k]) <= threshold_sigma * stds[k]
        outlier_mask_values &= ~in_cluster  # если попал хоть в один - не выброс

    n_outliers = outlier_mask_values.sum()

    # Применяем к оригинальной серии
    result = series.copy()
    valid_indices = series[valid_mask].index
    result.loc[valid_indices[outlier_mask_values]] = np.nan

    logger.info(f"    GMM {col_name}: {n_outliers} выбросов "
                f"(не попали ни в один кластер при {threshold_sigma}σ)")

    return result, n_outliers, cluster_info


def clean_attributes_a(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Очистка атрибутов Источника А.
    Двухступенчатая фильтрация:
      Ступень 1: жесткие пороги - убираем явный мусор (≤0, >90 этажей)
      Ступень 2: GMM - на оставшихся данных ищем бимодальные кластеры,
                 выбросы = значения, не принадлежащие ни одному кластеру
    """
    stats = {}

    for col in ['gkh_floor_count_min', 'gkh_floor_count_max']:
        if col not in gdf.columns:
            continue

        # Ступень 1: жесткие пороги
        mask_hard = (gdf[col] <= 0) | (gdf[col] > 90)
        n_hard = mask_hard.sum()
        gdf.loc[mask_hard, col] = np.nan
        logger.info(f"  {col} ступень 1 (жесткие пороги): {n_hard} выбросов (≤0 или >90)")

        # Ступень 2: GMM на очищенных
        gdf[col], n_gmm, cl_info = gmm_outlier_filter(
            gdf[col], n_components=2, threshold_sigma=3.0, col_name=col
        )
        stats[f'{col}_hard_outliers'] = n_hard
        stats[f'{col}_gmm_outliers'] = n_gmm
        stats[f'{col}_clusters'] = cl_info

    # Проверка: min не может быть больше max
    if 'gkh_floor_count_min' in gdf.columns and 'gkh_floor_count_max' in gdf.columns:
        mask = (gdf['gkh_floor_count_min'].notna() &
                gdf['gkh_floor_count_max'].notna() &
                (gdf['gkh_floor_count_min'] > gdf['gkh_floor_count_max']))
        n_swap = mask.sum()
        if n_swap > 0:
            gdf.loc[mask, ['gkh_floor_count_min', 'gkh_floor_count_max']] = \
                gdf.loc[mask, ['gkh_floor_count_max', 'gkh_floor_count_min']].values
            logger.info(f"  Исправлено min > max этажности: {n_swap}")

    return gdf, stats


def clean_attributes_b(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Очистка атрибутов Источника Б.
    Двухступенчатая фильтрация:
      Ступень 1: жесткие пороги - убираем явный мусор
        height: <= 0 или > 500 м (Лахта Центр = 462 м)
        stairs: <= 0 или > 90 (Лахта Центр ≈ 87 этажей)
      Ступень 2: GMM - бимодальная фильтрация на оставшихся данных
    """
    stats = {}

    # height
    if 'height' in gdf.columns:
        # Ступень 1
        mask_hard = (gdf['height'] <= 0) | (gdf['height'] > 500)
        n_hard = mask_hard.sum()
        gdf.loc[mask_hard, 'height'] = np.nan
        logger.info(f"  height ступень 1: {n_hard} выбросов (≤0 или >500 м)")

        # Ступень 2
        gdf['height'], n_gmm, cl_info = gmm_outlier_filter(
            gdf['height'], n_components=2, threshold_sigma=3.0, col_name='height'
        )
        stats['height_hard_outliers'] = n_hard
        stats['height_gmm_outliers'] = n_gmm
        stats['height_clusters'] = cl_info

    # stairs
    if 'stairs' in gdf.columns:
        # Ступень 1
        mask_hard = (gdf['stairs'] <= 0) | (gdf['stairs'] > 90)
        n_hard = mask_hard.sum()
        gdf.loc[mask_hard, 'stairs'] = np.nan
        logger.info(f"  stairs ступень 1: {n_hard} выбросов (≤0 или >90)")

        # Ступень 2
        gdf['stairs'], n_gmm, cl_info = gmm_outlier_filter(
            gdf['stairs'], n_components=2, threshold_sigma=3.0, col_name='stairs'
        )
        stats['stairs_hard_outliers'] = n_hard
        stats['stairs_gmm_outliers'] = n_gmm
        stats['stairs_clusters'] = cl_info

    # Согласованность: height vs stairs * avg_floor_height
    if all(c in gdf.columns for c in ['height', 'stairs', 'avg_floor_height']):
        mask = (gdf['height'].notna() & gdf['stairs'].notna() &
                gdf['avg_floor_height'].notna() & (gdf['avg_floor_height'] > 0))
        subset = gdf.loc[mask]

        if len(subset) > 0:
            expected = subset['stairs'] * subset['avg_floor_height']
            actual = subset['height']
            rel_error = np.abs(actual - expected) / expected

            # Расхождение > 100% - помечаем как подозрительное
            suspicious = rel_error > 1.0
            n_susp = suspicious.sum()
            gdf['height_suspicious'] = False
            gdf.loc[subset.index[suspicious], 'height_suspicious'] = True
            stats['height_suspicious'] = n_susp
            logger.info(f"  Несогласованность height vs stairs*avg_floor_height: {n_susp}")

            # Расхождение > 300% - обнуляем height (скорее всего ошибка)
            very_wrong = rel_error > 3.0
            n_reset = very_wrong.sum()
            if n_reset > 0:
                gdf.loc[subset.index[very_wrong], 'height'] = np.nan
                logger.info(f"  Сброшено height с ошибкой > 300%: {n_reset}")

    return gdf, stats

# ФИЛЬТРАЦИЯ МУСОРНЫХ ОБЪЕКТОВ

def filter_tiny_objects(gdf: gpd.GeoDataFrame, min_area_sq_m: float = 10) -> gpd.GeoDataFrame:
    """
    Удаление объектов с площадью < min_area_sq_m.
    Площадь считается в метрической проекции (UTM 36N).

    10 м² - это примерно 3×3 м. Такие объекты - сараи, будки, гаражи.
    Для зданий это не действительно.
    """
    gdf_proj = gdf.to_crs(METRIC_CRS)
    area_m2 = gdf_proj.geometry.area

    # Сохраняем площадь для дальнейшего использования
    gdf['area_m2_calc'] = area_m2.values

    before = len(gdf)
    gdf = gdf[area_m2 >= min_area_sq_m].copy()
    removed = before - len(gdf)

    if removed > 0:
        logger.info(f"  Удалено мелких объектов (< {min_area_sq_m} м²): {removed}")

    return gdf

# ГЛАВНЫЕ ФУНКЦИИ ОЧИСТКИ

def clean_source_a(gdf: gpd.GeoDataFrame) -> tuple:
    """Полная очистка Источника А."""
    logger.info("=" * 60)
    logger.info("ОЧИСТКА ИСТОЧНИКА А")
    logger.info("=" * 60)

    all_stats = {'source': 'A', 'initial': len(gdf)}
    t0 = time.time()

    # 1. Геометрия (типы 1, 4, 5, 6, 7)
    logger.info("\n Исправление геометрии ")
    gdf, geom_stats = fix_all_geometry(gdf)
    all_stats.update(geom_stats)

    # 2. Удаление точных дубликатов (побайтовое совпадение геометрии)
    logger.info("\n Удаление дубликатов ")
    gdf = remove_exact_duplicates(gdf, 'A')
    all_stats['after_dedup'] = len(gdf)

    # 3. Очистка атрибутов
    logger.info("\n Очистка атрибутов ")
    gdf, attr_stats = clean_attributes_a(gdf)
    all_stats.update(attr_stats)

    # 4. Фильтрация мусора
    logger.info("\n Фильтрация мелких объектов ")
    gdf = filter_tiny_objects(gdf, min_area_sq_m=10)
    all_stats['final'] = len(gdf)

    gdf = gdf.reset_index(drop=True)

    elapsed = time.time() - t0
    removed_total = all_stats['initial'] - all_stats['final']
    pct = removed_total / all_stats['initial'] * 100 if all_stats['initial'] > 0 else 0

    logger.info(f"\n  ИТОГО А: {all_stats['initial']} - {all_stats['final']} "
                f"(убрано {removed_total}, {pct:.1f}%) за {elapsed:.1f}с")

    return gdf, all_stats


def clean_source_b(gdf: gpd.GeoDataFrame) -> tuple:
    """Полная очистка Источника Б."""
    logger.info("=" * 60)
    logger.info("ОЧИСТКА ИСТОЧНИКА Б")
    logger.info("=" * 60)

    all_stats = {'source': 'B', 'initial': len(gdf)}
    t0 = time.time()

    # 1. Геометрия
    logger.info("\n Исправление геометрии ")
    gdf, geom_stats = fix_all_geometry(gdf)
    all_stats.update(geom_stats)

    # 2. Удаление точных дубликатов
    logger.info("\n Удаление дубликатов ")
    gdf = remove_exact_duplicates(gdf, 'B')
    all_stats['after_dedup'] = len(gdf)

    # 3. Очистка атрибутов
    logger.info("\n Очистка атрибутов ")
    gdf, attr_stats = clean_attributes_b(gdf)
    all_stats.update(attr_stats)

    # 4. Фильтрация мусора
    logger.info("\n Фильтрация мелких объектов ")
    gdf = filter_tiny_objects(gdf, min_area_sq_m=10)
    all_stats['final'] = len(gdf)

    gdf = gdf.reset_index(drop=True)

    elapsed = time.time() - t0
    removed_total = all_stats['initial'] - all_stats['final']
    pct = removed_total / all_stats['initial'] * 100 if all_stats['initial'] > 0 else 0

    logger.info(f"\n  ИТОГО Б: {all_stats['initial']} - {all_stats['final']} "
                f"(убрано {removed_total}, {pct:.1f}%) за {elapsed:.1f}с")

    return gdf, all_stats

# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

def save_clean_data(gdf: gpd.GeoDataFrame, filepath: str):
    """Сохранение очищенных данных в CSV с WKT-геометрией."""
    df = gdf.copy()
    df['geometry_wkt'] = df.geometry.apply(lambda g: g.wkt if g is not None else None)
    df = df.drop(columns=['geometry'])
    df.to_csv(filepath, index=False)
    logger.info(f"  Сохранено: {filepath} ({len(df)} записей)")

# ТОЧКА ВХОДА

if __name__ == '__main__':
    import sys

    SOURCE_A = 'data/cup_it_example_src_A.csv'
    SOURCE_B = 'data/cup_it_example_src_B.csv'
    OUTPUT_DIR = 'output'

    if len(sys.argv) >= 3:
        SOURCE_A = sys.argv[1]
        SOURCE_B = sys.argv[2]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ЗАПУСК ОЧИСТКИ ДАННЫХ")
    logger.info("=" * 60)
    t_total = time.time()

    #  Загрузка
    gdf_a = load_source(SOURCE_A, "Источник А")
    gdf_b = load_source(SOURCE_B, "Источник Б")

    #  Очистка
    gdf_a_clean, stats_a = clean_source_a(gdf_a)
    gdf_b_clean, stats_b = clean_source_b(gdf_b)

    #  Сохранение
    logger.info("\n" + "=" * 60)
    logger.info("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)

    save_clean_data(gdf_a_clean, os.path.join(OUTPUT_DIR, 'clean_A.csv'))
    save_clean_data(gdf_b_clean, os.path.join(OUTPUT_DIR, 'clean_B.csv'))

    # Сохраняем статистику
    stats_all = {'source_a': stats_a, 'source_b': stats_b}
    with open(os.path.join(OUTPUT_DIR, 'cleaning_stats.json'), 'w') as f:
        json.dump(stats_all, f, indent=2, default=str)
    logger.info(f"  Статистика: {OUTPUT_DIR}/cleaning_stats.json")

    elapsed = time.time() - t_total
    logger.info(f"\n ОЧИСТКА ЗАВЕРШЕНА за {elapsed:.1f}с")

    # Итоговая сводка
    logger.info("\n" + "=" * 60)
    logger.info("СВОДКА")
    logger.info("=" * 60)
    logger.info(f"  А: {stats_a['initial']} - {stats_a['final']} "
                f"(убрано {stats_a['initial'] - stats_a['final']})")
    logger.info(f"  Б: {stats_b['initial']} - {stats_b['final']} "
                f"(убрано {stats_b['initial'] - stats_b['final']})")