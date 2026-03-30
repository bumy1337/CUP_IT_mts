"""

Скрипт: Обогащение данных из OpenStreetMap

ЧТО ДЕЛАЕТ:
           
1. Скачивает данные о зданиях Питера из OpenStreetMap (Overpass API)
2. Извлекает building:levels (этажность) - есть примерно для 30-40% зданий
3. Пространственно сопоставляет OSM-здания с нашими через IoU
4. Добавляет OSM-этажность как дополнительный сигнал для валидации
5. Для конфликтных зданий - третий голос для разрешения

ЗАЧЕМ ЭТО НУЖНО:
                 
Сейчас у нас 2 источника. OSM - это ТРЕТИЙ НЕЗАВИСИМЫЙ источник.
Три источника > двух:
  - Голосование становится надежнее (медиана из 3 = устойчива к 1 выбросу)
  - Перекрестная валидация: если все 3 согласны - уверенность максимальна
  - Для 1007 конфликтных зданий - OSM может быть решающим голосом

ОГРАНИЧЕНИЯ:
            
- OSM данные заполнены волонтерами, не все здания имеют этажность
- Качество зависит от региона (Питер - хорошо покрыт)
- Геометрия в OSM может отличаться от наших источников
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import shape, Polygon, MultiPolygon
import requests
import warnings
import logging
import time
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRIC_CRS = 'EPSG:32636'

# Границы Санкт-Петербурга (bounding box)
SPB_BBOX = {
    'south': 59.75,
    'west': 29.40,
    'north': 60.15,
    'east': 30.75,
}

# ЭТАП 1: СКАЧИВАНИЕ ДАННЫХ ИЗ OSM

def download_osm_buildings(bbox=SPB_BBOX, cache_file='output/osm_buildings_cache.json'):
    """
    Скачивает здания с тегом building:levels из OpenStreetMap.
    Пробует несколько серверов Overpass API если основной недоступен.
    """
    if os.path.exists(cache_file):
        logger.info(f"  Загрузка из кэша: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    logger.info("  Скачивание зданий из OpenStreetMap...")

    query = f"""
    [out:json][timeout:600];
    (
      way["building"]["building:levels"]
        ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out body;
    >;
    out skel qt;
    """

    # Несколько серверов Overpass (если один упал - пробуем другой)
    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for server_url in servers:
        logger.info(f"  Попытка: {server_url}")
        try:
            t0 = time.time()
            response = requests.post(
                server_url,
                data={'data': query},
                timeout=900,
                headers={'User-Agent': 'MTS-CupIT-2026/1.0'}
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - t0
            logger.info(f"  Получено за {elapsed:.1f}с")

            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"  Сохранено в кэш: {cache_file}")
            return data

        except Exception as e:
            logger.warning(f"  Ошибка: {e}")
            logger.info("  Пробуем следующий сервер...")
            time.sleep(5)

    logger.error("  Все серверы недоступны!")
    return None

# ЭТАП 2: ПАРСИНГ OSM ДАННЫХ В GEODATAFRAME

def parse_osm_to_gdf(osm_data):
    """
    Превращает сырой JSON из Overpass в GeoDataFrame.

    OSM хранит данные так:
      - Nodes (точки): id, lat, lon
      - Ways (линии/полигоны): id, список node_ids, теги
      - Relations (сложные объекты): id, список members, теги

    Нам нужны ways с тегом building:levels.
    Мы собираем координаты nodes, формируем полигоны.
    """
    if osm_data is None:
        return None

    logger.info("  Парсинг OSM данных...")

    elements = osm_data.get('elements', [])

    # Собираем все nodes (точки с координатами)
    nodes = {}
    for el in elements:
        if el['type'] == 'node':
            nodes[el['id']] = (el['lon'], el['lat'])

    # Собираем ways с building:levels
    buildings = []
    for el in elements:
        if el['type'] != 'way':
            continue
        tags = el.get('tags', {})
        levels = tags.get('building:levels')
        if levels is None:
            continue

        # Пробуем распарсить этажность
        try:
            levels = float(levels)
            if levels <= 0 or levels > 100:
                continue
        except (ValueError, TypeError):
            continue

        # Собираем координаты полигона
        coords = []
        for node_id in el.get('nodes', []):
            if node_id in nodes:
                coords.append(nodes[node_id])

        if len(coords) < 4:  # полигон минимум 4 точки (3 + замыкающая)
            continue

        # Замыкаем если нужно
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                buildings.append({
                    'osm_id': el['id'],
                    'osm_levels': levels,
                    'osm_height_est': levels * 3.0,  # оценка высоты
                    'osm_building_type': tags.get('building', 'yes'),
                    'geometry': poly,
                })
        except Exception:
            continue

    if not buildings:
        logger.warning("  Не удалось распарсить ни одного здания из OSM")
        return None

    gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs='EPSG:4326')
    logger.info(f"  Распарсено зданий из OSM: {len(gdf)}")
    logger.info(f"  Этажность: min={gdf['osm_levels'].min()}, "
                f"max={gdf['osm_levels'].max()}, "
                f"median={gdf['osm_levels'].median()}")

    return gdf

# ЭТАП 3: СОПОСТАВЛЕНИЕ OSM С НАШИМИ ЗДАНИЯМИ

def match_osm_to_buildings(buildings, osm_gdf, gdf_a, gdf_b, components):
    """
    Пространственно сопоставляет OSM-здания с нашими через центроиды + площадь.

    Для каждого нашего здания:
      1. Ищем OSM-здания чей центроид < 20 м
      2. Проверяем соотношение площадей < 3×
      3. Если совпало - записываем osm_levels
    """
    logger.info("\n  Сопоставление OSM с нашими зданиями...")

    # Собираем геометрии наших зданий
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
                if g is not None and not g.is_empty:
                    geom = g
                    break
        geom_list.append(geom)

    our_gdf = gpd.GeoDataFrame(
        buildings.copy(),
        geometry=geom_list,
        crs='EPSG:4326'
    )

    # Проецируем в метры
    our_proj = our_gdf.to_crs(METRIC_CRS)
    osm_proj = osm_gdf.to_crs(METRIC_CRS)

    our_centroids = our_proj.geometry.centroid
    our_areas = our_proj.geometry.area
    osm_centroids = osm_proj.geometry.centroid
    osm_areas = osm_proj.geometry.area

    # Пространственный индекс OSM
    osm_sindex = osm_proj.sindex

    osm_levels_col = np.full(len(buildings), np.nan)
    osm_height_col = np.full(len(buildings), np.nan)
    osm_matched = 0

    for i in range(len(our_proj)):
        c = our_centroids.iloc[i]
        if c is None or c.is_empty:
            continue

        our_area = our_areas.iloc[i]
        if our_area <= 0:
            continue

        # Ищем OSM-кандидатов в радиусе 20 м
        buf = c.buffer(20)
        cands = list(osm_sindex.intersection(buf.bounds))

        best_dist = 999
        best_j = None

        for j in cands:
            dist = c.distance(osm_centroids.iloc[j])
            if dist > 20:
                continue

            # Проверяем площадь
            osm_area = osm_areas.iloc[j]
            if osm_area > 0 and our_area > 0:
                ratio = max(osm_area, our_area) / min(osm_area, our_area)
                if ratio > 3:
                    continue

            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            osm_levels_col[i] = osm_gdf.iloc[best_j]['osm_levels']
            osm_height_col[i] = osm_gdf.iloc[best_j]['osm_height_est']
            osm_matched += 1

        if (i + 1) % 50000 == 0:
            logger.info(f"    {i+1}/{len(our_proj)}, совпадений: {osm_matched}")

    buildings['osm_levels'] = osm_levels_col
    buildings['osm_height_est'] = osm_height_col

    n_with_osm = buildings['osm_levels'].notna().sum()
    logger.info(f"\n  Совпадений с OSM: {osm_matched}")
    logger.info(f"  Зданий с OSM-этажностью: {n_with_osm}")

    return buildings, osm_matched

# ЭТАП 4: ИСПОЛЬЗОВАНИЕ OSM ДЛЯ ВАЛИДАЦИИ И КОРРЕКЦИИ

def apply_osm_validation(buildings):
    """
    Использует OSM как третий голос:
      1. Для зданий с OSM - сравниваем нашу высоту с OSM-оценкой
      2. Если расхождение < 30% - повышаем confidence
      3. Для конфликтных зданий - OSM = решающий голос
    """
    logger.info("\n  Применение OSM для валидации...")

    has_osm = buildings['osm_height_est'].notna()
    n_osm = has_osm.sum()

    if n_osm == 0:
        logger.info("  Нет OSM данных для валидации")
        return buildings, {}

    # Сравниваем нашу высоту с OSM
    both = buildings[has_osm & buildings['height_final'].notna()].copy()
    both['osm_diff'] = np.abs(both['height_final'] - both['osm_height_est'])
    both['osm_rel_diff'] = both['osm_diff'] / (both['height_final'] + 0.1)

    agree_1floor = (both['osm_diff'] < 3).sum()
    agree_2floors = (both['osm_diff'] < 6).sum()
    total_compared = len(both)

    stats = {
        'osm_buildings_total': int(n_osm),
        'compared_with_our': total_compared,
        'agree_within_1_floor': int(agree_1floor),
        'agree_within_1_floor_pct': round(agree_1floor / max(total_compared, 1) * 100, 1),
        'agree_within_2_floors': int(agree_2floors),
        'agree_within_2_floors_pct': round(agree_2floors / max(total_compared, 1) * 100, 1),
        'mean_diff_m': round(both['osm_diff'].mean(), 2),
        'median_diff_m': round(both['osm_diff'].median(), 2),
    }

    logger.info(f"  Сравнено с OSM: {total_compared} зданий")
    logger.info(f"  Согласованы < 1 этажа: {agree_1floor} ({stats['agree_within_1_floor_pct']}%)")
    logger.info(f"  Согласованы < 2 этажей: {agree_2floors} ({stats['agree_within_2_floors_pct']}%)")
    logger.info(f"  Среднее расхождение: {stats['mean_diff_m']} м")

    # Повышаем confidence для согласованных с OSM
    osm_agrees = has_osm & buildings['height_final'].notna()
    osm_diff = np.abs(buildings['height_final'] - buildings['osm_height_est'])

    upgrade_mask = osm_agrees & (osm_diff < 3)
    n_upgraded = upgrade_mask.sum()
    buildings.loc[upgrade_mask, 'height_confidence'] = (
        buildings.loc[upgrade_mask, 'height_confidence'] + '_osm_confirmed'
    )

    logger.info(f"  Подтверждены OSM (< 3 м): {n_upgraded} зданий")

    # Для конфликтных - используем OSM как решающий голос
    conflict_with_osm = (
        (buildings['height_confidence'] == 'low_conflicting') &
        has_osm
    )
    n_conflict_resolved = conflict_with_osm.sum()
    if n_conflict_resolved > 0:
        buildings.loc[conflict_with_osm, 'height_final'] = (
            buildings.loc[conflict_with_osm, 'osm_height_est']
        )
        buildings.loc[conflict_with_osm, 'height_source'] = 'resolved_by_osm'
        buildings.loc[conflict_with_osm, 'height_confidence'] = 'resolved_by_osm'
        logger.info(f"  Конфликты разрешены через OSM: {n_conflict_resolved}")

    stats['confidence_upgraded'] = int(n_upgraded)
    stats['conflicts_resolved_by_osm'] = int(n_conflict_resolved)

    return buildings, stats

# ТОЧКА ВХОДА

if __name__ == '__main__':
    OUTPUT_DIR = 'output'
    BUILDINGS = os.path.join(OUTPUT_DIR, 'buildings_with_height.csv')
    CLEAN_A = os.path.join(OUTPUT_DIR, 'clean_A.csv')
    CLEAN_B = os.path.join(OUTPUT_DIR, 'clean_B.csv')
    COMPONENTS = os.path.join(OUTPUT_DIR, 'matched_components.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ОБОГАЩЕНИЕ ДАННЫХ ИЗ OPENSTREETMAP")
    logger.info("=" * 60)
    t0 = time.time()

    # Загрузка наших данных
    buildings = pd.read_csv(BUILDINGS)
    logger.info(f"Наши здания: {len(buildings)}")

    # Загрузка геоданных для сопоставления
    def load_gdf(path, name):
        df = pd.read_csv(path)
        gcol = None
        for c in ['geometry_wkt', 'geometry', 'wkt']:
            if c in df.columns: gcol = c; break
        df['geometry'] = df[gcol].apply(lambda x: wkt.loads(x) if pd.notna(x) and isinstance(x, str) else None)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        s = gdf.geometry.dropna().iloc[0].centroid
        gdf = gdf.set_crs('EPSG:4326' if -180 <= s.x <= 180 else METRIC_CRS)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    gdf_a = load_gdf(CLEAN_A, "А")
    gdf_b = load_gdf(CLEAN_B, "Б")
    components = pd.read_csv(COMPONENTS)

    # Этап 1: Скачивание OSM
    logger.info("\n  Этап 1: Скачивание OSM  ")
    osm_data = download_osm_buildings()

    if osm_data is None:
        logger.error("Не удалось получить OSM данные. Выход.")
        exit(1)

    # Этап 2: Парсинг
    logger.info("\n  Этап 2: Парсинг OSM  ")
    osm_gdf = parse_osm_to_gdf(osm_data)

    if osm_gdf is None or len(osm_gdf) == 0:
        logger.error("Нет данных после парсинга. Выход.")
        exit(1)

    # Этап 3: Сопоставление
    logger.info("\n  Этап 3: Сопоставление с нашими зданиями  ")
    buildings, n_matched = match_osm_to_buildings(buildings, osm_gdf, gdf_a, gdf_b, components)

    # Этап 4: Валидация и коррекция
    logger.info("\n  Этап 4: Валидация через OSM ")
    buildings, osm_stats = apply_osm_validation(buildings)

    # Сохранение
    logger.info("\n" + "=" * 60)
    logger.info("СОХРАНЕНИЕ")
    output_path = os.path.join(OUTPUT_DIR, 'buildings_with_osm.csv')
    buildings.to_csv(output_path, index=False)
    logger.info(f"  Результат: {output_path}")

    with open(os.path.join(OUTPUT_DIR, 'osm_stats.json'), 'w') as f:
        json.dump(osm_stats, f, indent=2, default=str)
    logger.info(f"  Статистика: {OUTPUT_DIR}/osm_stats.json")

    logger.info(f"\n OSM ОБОГАЩЕНИЕ ЗАВЕРШЕНО за {time.time()-t0:.1f}с")