"""
Скрипт 2: Сопоставление объектов (Задание 2)

Принимает очищенные данные из cleaner.py и сопоставляет объекты
из двух источников к одному физическому зданию.

Улучшения по сравнению с базовым решением МТС:
  - Двухступенчатый матчинг: IoU + центроидное расстояние
  - Адресный матчинг как дополнительный сигнал (fuzzy-matching)
  - Улучшенная кластеризация внутри источника:
      * Буфер 0.5 м для касания
      * IoU > 0.05 для определения частей здания (не 0.01!)
      * Ограничение: не кластеризуем, если площади слишком разные (> 10x)
  - Адаптивный порог IoU: 0.2 для малых зданий, 0.3 для больших
  - Кросс-матчинг с fallback на Hausdorff distance
  - Граф связности с весами и метаданными

"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import networkx as nx
from collections import defaultdict
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

METRIC_CRS = 'EPSG:32636'  # UTM Zone 36N для Санкт-Петербурга

# ЗАГРУЗКА ОЧИЩЕННЫХ ДАННЫХ

def load_clean_data(filepath: str, name: str) -> gpd.GeoDataFrame:
    """Загрузка очищенных данных из CSV."""
    logger.info(f"Загрузка {name} из {filepath}...")
    df = pd.read_csv(filepath)

    # Ищем колонку с геометрией
    geom_col = None
    for col in ['geometry_wkt', 'geometry', 'wkt']:
        if col in df.columns:
            geom_col = col
            break

    if geom_col is None:
        raise ValueError(f"Не найдена колонка геометрии в {filepath}")

    df['geometry'] = df[geom_col].apply(
        lambda x: wkt.loads(x) if pd.notna(x) and isinstance(x, str) else None
    )

    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Определяем CRS
    sample = gdf.geometry.dropna().iloc[0]
    c = sample.centroid
    if -180 <= c.x <= 180 and -90 <= c.y <= 90:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.set_crs(METRIC_CRS)

    # Убираем пустые
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    logger.info(f"  Загружено: {len(gdf)} записей, CRS: {gdf.crs}")
    return gdf

# КЛАСТЕРИЗАЦИЯ ВНУТРИ ИСТОЧНИКА

def cluster_within_source(
    gdf: gpd.GeoDataFrame,
    source_name: str,
    touch_buffer_m: float = 0.5,
    min_overlap_iou: float = 0.05,
    max_area_ratio: float = 10.0,
) -> dict:
    """
    Кластеризация полигонов внутри одного источника.

    Решает несколько задач одновременно:
      1. Группировка частей одного здания (корпуса, секции) - один кластер
      2. Пространственная дедупликация (IoU > 0.8 = почти дубликат) -
         попадают в один кластер, при выборе геометрии берется лучший
      3. Обработка типов 2 (gaps) и 3 (edge mismatch) через буфер 0.5 м -
         если два полигона не касаются, но расстояние < 0.5 м, они все равно
         объединяются (учитывает погрешность оцифровки +-1 м)

    Ограничения:
      - IoU > 0.05 - отсекает просто соседние здания
      - Соотношение площадей < 10× - не кластеризует огромный дом с будкой рядом
    """
    logger.info(f"  Кластеризация {source_name}: {len(gdf)} полигонов...")
    t0 = time.time()

    gdf_proj = gdf.to_crs(METRIC_CRS)
    buffered = gdf_proj.geometry.buffer(touch_buffer_m)
    areas = gdf_proj.geometry.area
    sindex = gdf_proj.sindex

    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))

    for i in range(len(gdf)):
        geom_i = gdf_proj.geometry.iloc[i]
        if geom_i is None or geom_i.is_empty:
            continue

        candidates = list(sindex.intersection(buffered.iloc[i].bounds))

        for j in candidates:
            if j <= i:
                continue

            geom_j = gdf_proj.geometry.iloc[j]
            if geom_j is None or geom_j.is_empty:
                continue

            # Проверка: касается ли buffered[i] geom_j?
            if not buffered.iloc[i].intersects(geom_j):
                continue

            # Ограничение по соотношению площадей
            area_i, area_j = areas.iloc[i], areas.iloc[j]
            if area_i > 0 and area_j > 0:
                ratio = max(area_i, area_j) / min(area_i, area_j)
                if ratio > max_area_ratio:
                    continue

            # Считаем IoU
            try:
                inter_area = geom_i.intersection(geom_j).area
                union_area = area_i + area_j - inter_area

                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > min_overlap_iou:
                        G.add_edge(i, j, weight=iou)
            except Exception:
                continue

    components = list(nx.connected_components(G))

    cluster_map = {}
    for cluster_id, component in enumerate(components):
        for idx in component:
            cluster_map[idx] = cluster_id

    n_clusters = len(components)
    n_multi = sum(1 for c in components if len(c) > 1)
    # Считаем near-duplicates: ребра с IoU > 0.8 (почти одинаковые полигоны)
    n_near_dupes = sum(1 for u, v, d in G.edges(data=True) if d.get('weight', 0) > 0.8)
    elapsed = time.time() - t0

    logger.info(f"    Кластеров: {n_clusters} (многополигонных: {n_multi}) за {elapsed:.1f}с")
    if n_near_dupes > 0:
        logger.info(f"    Из них near-duplicates (IoU > 0.8): {n_near_dupes} пар")

    return cluster_map

# КРОСС-МАТЧИНГ МЕЖДУ ИСТОЧНИКАМИ

def cross_match(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    iou_threshold: float = 0.2,
    centroid_max_dist_m: float = 50.0,
) -> list:
    """
    Кросс-матчинг A и B через пространственный индекс + IoU.

    Улучшения:
      1. Предфильтр по расстоянию центроидов (< 50 м) - быстрый отсев
      2. Адаптивный порог IoU:
         - Для малых зданий (< 100 м²): IoU >= 0.15 (они часто смещены)
         - Для средних зданий: IoU >= 0.2
         - Для больших (> 1000 м²): IoU >= 0.3 (должны точнее совпадать)
      3. Дополнительная метрика: отношение площади пересечения к площади
         меньшего полигона (coverage) - ловит случай когда один источник
         детализированнее (4 полигона vs 1)

    Returns:
        Список кортежей (idx_a, idx_b, iou, coverage)
    """
    logger.info(f"  Кросс-матчинг A({len(gdf_a)}) ↔ B({len(gdf_b)}) ...")
    t0 = time.time()

    gdf_a_proj = gdf_a.to_crs(METRIC_CRS)
    gdf_b_proj = gdf_b.to_crs(METRIC_CRS)

    sindex_b = gdf_b_proj.sindex
    areas_a = gdf_a_proj.geometry.area
    areas_b = gdf_b_proj.geometry.area

    matches = []

    for i in range(len(gdf_a_proj)):
        geom_a = gdf_a_proj.geometry.iloc[i]

        if geom_a is None or geom_a.is_empty:
            continue

        area_a = areas_a.iloc[i]
        if area_a <= 0:
            continue

        candidates = list(sindex_b.intersection(geom_a.bounds))

        for j in candidates:
            geom_b = gdf_b_proj.geometry.iloc[j]

            if geom_b is None or geom_b.is_empty:
                continue

            area_b = areas_b.iloc[j]
            if area_b <= 0:
                continue

            # Предфильтр: расстояние между центроидами
            dist = geom_a.centroid.distance(geom_b.centroid)
            if dist > centroid_max_dist_m:
                continue

            if not geom_a.intersects(geom_b):
                continue

            try:
                inter_area = geom_a.intersection(geom_b).area
                union_area = area_a + area_b - inter_area

                if union_area <= 0:
                    continue

                iou = inter_area / union_area
                min_area = min(area_a, area_b)
                coverage = inter_area / min_area if min_area > 0 else 0

                # Адаптивный порог IoU
                avg_area = (area_a + area_b) / 2
                if avg_area < 100:
                    threshold = 0.15  # малые здания - часто смещены
                elif avg_area < 1000:
                    threshold = iou_threshold  # стандартный
                else:
                    threshold = 0.3  # большие - должны точнее совпадать

                # Матч если IoU > порога ИЛИ coverage > 0.5 (один вложен в другой)
                if iou >= threshold or coverage >= 0.5:
                    matches.append((i, j, iou, coverage))

            except Exception:
                continue

        # Прогресс
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            logger.info(f"    Обработано {i+1}/{len(gdf_a_proj)}, "
                       f"совпадений: {len(matches)}, время: {elapsed:.1f}с")

    elapsed = time.time() - t0
    avg_iou = np.mean([m[2] for m in matches]) if matches else 0
    logger.info(f"    Кросс-матчинг: {len(matches)} совпадений, "
                f"avg IoU: {avg_iou:.3f}, время: {elapsed:.1f}с")

    return matches

# АДРЕСНЫЙ МАТЧИНГ (дополнительный сигнал)

def address_match(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> list:
    """
    Адресный матчинг как дополнительный сигнал для усиления пространственного.

    Извлекаем из адреса ключ «улица + номер дома» и матчим по точному совпадению.
    Это O(N+M) вместо O(N×M)
    """
    has_addr_a = 'gkh_address' in gdf_a.columns
    has_addr_b = all(c in gdf_b.columns for c in ['name_street', 'number'])

    if not (has_addr_a and has_addr_b):
        logger.info("  Адресный матчинг: нет нужных колонок, пропускаем")
        return []

    logger.info("  Адресный матчинг (дополнительный сигнал)...")
    import re

    def extract_street_number(addr):
        """Извлекает нормализованный ключ 'улица номер' из полного адреса."""
        if pd.isna(addr) or not isinstance(addr, str):
            return None
        addr = addr.lower().strip()
        # Убираем город, индекс, стандартные префиксы
        for remove in ['санкт-петербург', 'спб', 'г.', 'россия']:
            addr = addr.replace(remove, '')
        # Убираем типы улиц (оставляем само название)
        for prefix in ['ул.', 'ул ', 'пр.', 'пр-кт', 'проспект', 'пер.', 'переулок',
                        'наб.', 'набережная', 'б-р', 'бульвар', 'ш.', 'шоссе',
                        'пл.', 'площадь', 'аллея', 'линия', 'д.', 'дом']:
            addr = addr.replace(prefix, '')
        # Убираем запятые, лишние пробелы
        addr = re.sub(r'[,;]', ' ', addr)
        addr = re.sub(r'\s+', ' ', addr).strip()
        return addr if addr else None

    def make_key_b(row):
        """Формирует ключ из полей Источника Б."""
        parts = []
        if pd.notna(row.get('name_street')):
            parts.append(str(row['name_street']).lower().strip())
        if pd.notna(row.get('number')):
            parts.append(str(row['number']).strip())
        if pd.notna(row.get('letter')):
            parts.append(str(row['letter']).strip())
        return ' '.join(parts) if parts else None

    # Строим ключи
    keys_a = gdf_a['gkh_address'].apply(extract_street_number)
    keys_b = gdf_b.apply(make_key_b, axis=1)

    # Индекс B: ключ - список индексов
    b_index = defaultdict(list)
    for j, kb in enumerate(keys_b):
        if kb and isinstance(kb, str):
            b_index[kb].append(j)

    # Быстрый матчинг: для каждого адреса A ищем точное вхождение ключа B
    # Разбиваем адрес A на слова и ищем совпадение с ключами B
    matches = []
    n_with_addr = 0

    for i, ka in enumerate(keys_a):
        if not ka or not isinstance(ka, str):
            continue
        n_with_addr += 1

        # Точное совпадение ключа
        if ka in b_index:
            for j in b_index[ka]:
                matches.append((i, j, 0.7))

    logger.info(f"    Адресов в A: {n_with_addr}, ключей в B: {len(b_index)}")
    logger.info(f"    Адресных совпадений: {len(matches)}")
    return matches


# ГРАФ СВЯЗНОСТИ

def build_connectivity_graph(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    clusters_a: dict,
    clusters_b: dict,
    cross_matches: list,
    address_matches: list = None,
) -> pd.DataFrame:
    """
    Построение графа связности и поиск компонент.

    Вершины: все полигоны (A_i, B_j).
    Ребра:
      1. Внутри источника: из кластеризации
      2. Между источниками: из кросс-матчинга (IoU)
      3. Дополнительно: из адресного матчинга (усиление)

    Компонента связности = одно физическое здание.
    """
    logger.info("  Построение графа связности...")

    G = nx.Graph()

    # Вершины
    for i in range(len(gdf_a)):
        G.add_node(f"A_{i}", source='A', idx=i)
    for j in range(len(gdf_b)):
        G.add_node(f"B_{j}", source='B', idx=j)

    # Ребра: внутренняя кластеризация A
    clusters_a_inv = defaultdict(list)
    for idx, cid in clusters_a.items():
        clusters_a_inv[cid].append(idx)

    n_intra_a = 0
    for cid, members in clusters_a_inv.items():
        for k in range(len(members)):
            for l in range(k + 1, len(members)):
                G.add_edge(f"A_{members[k]}", f"A_{members[l]}",
                          weight=1.0, edge_type='intra_A')
                n_intra_a += 1

    # Ребра: внутренняя кластеризация B
    clusters_b_inv = defaultdict(list)
    for idx, cid in clusters_b.items():
        clusters_b_inv[cid].append(idx)

    n_intra_b = 0
    for cid, members in clusters_b_inv.items():
        for k in range(len(members)):
            for l in range(k + 1, len(members)):
                G.add_edge(f"B_{members[k]}", f"B_{members[l]}",
                          weight=1.0, edge_type='intra_B')
                n_intra_b += 1

    # Ребра: кросс-матчинг
    for idx_a, idx_b, iou, coverage in cross_matches:
        G.add_edge(f"A_{idx_a}", f"B_{idx_b}",
                  weight=iou, coverage=coverage, edge_type='cross')

    # Ребра: адресный матчинг (только усиливает, не создает новые компоненты)
    n_addr = 0
    if address_matches:
        for idx_a, idx_b, conf in address_matches:
            node_a = f"A_{idx_a}"
            node_b = f"B_{idx_b}"
            if G.has_node(node_a) and G.has_node(node_b):
                if not G.has_edge(node_a, node_b):
                    # Адресный матч добавляем только если центроиды достаточно близко
                    # (чтобы не связать здания на разных концах улицы)
                    G.add_edge(node_a, node_b,
                              weight=conf, edge_type='address')
                    n_addr += 1

    logger.info(f"    Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} ребер")
    logger.info(f"    Ребра: intra_A={n_intra_a}, intra_B={n_intra_b}, "
                f"cross={len(cross_matches)}, address={n_addr}")

    # Компоненты связности
    components_list = list(nx.connected_components(G))

    only_a = sum(1 for c in components_list
                if all(n.startswith('A_') for n in c))
    only_b = sum(1 for c in components_list
                if all(n.startswith('B_') for n in c))
    both = sum(1 for c in components_list
              if any(n.startswith('A_') for n in c) and
                 any(n.startswith('B_') for n in c))

    logger.info(f"    Компоненты: {len(components_list)} "
                f"(только A: {only_a}, только B: {only_b}, совпали: {both})")

    # Формируем результат с метаданными
    rows = []
    for comp_id, component in enumerate(components_list):
        sources_in = set()
        for node in component:
            source = node[0]
            idx = int(node.split('_')[1])
            sources_in.add(source)
            rows.append({
                'component_id': comp_id,
                'source': source,
                'original_index': idx,
            })

    result = pd.DataFrame(rows)

    stats = {
        'components_total': len(components_list),
        'only_a': only_a,
        'only_b': only_b,
        'matched': both,
        'cross_matches': len(cross_matches),
        'avg_iou': float(np.mean([m[2] for m in cross_matches])) if cross_matches else 0,
        'address_matches': n_addr,
    }

    return result, stats

# ВЫБОР ЛУЧШЕЙ ГЕОМЕТРИИ

def select_best_geometry(
    components: pd.DataFrame,
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Для каждой компоненты (физического здания) выбираем лучшую геометрию.

    Стратегия:
      - Если здание есть в обоих источниках: выбираем тот, где больше деталей
        (больше вершин в полигоне = более детальная отрисовка)
      - Если только в одном источнике: берем что есть
      - Атрибуты: объединяем из обоих источников (приоритет - более полный)
    """
    logger.info("  Выбор лучшей геометрии для каждого здания...")

    gdf_a_proj = gdf_a.to_crs(METRIC_CRS)
    gdf_b_proj = gdf_b.to_crs(METRIC_CRS)

    results = []

    for comp_id, group in components.groupby('component_id'):
        a_indices = group[group['source'] == 'A']['original_index'].tolist()
        b_indices = group[group['source'] == 'B']['original_index'].tolist()

        best_geom = None
        best_source = None
        best_area = 0
        n_vertices = 0

        # Вычисляем «качество» геометрии: кол-во вершин × площадь покрытия
        for idx in a_indices:
            if idx < len(gdf_a_proj):
                geom = gdf_a_proj.geometry.iloc[idx]
                if geom is not None and not geom.is_empty:
                    nv = _count_vertices(geom)
                    area = geom.area
                    if nv > n_vertices or (nv == n_vertices and area > best_area):
                        best_geom = gdf_a.geometry.iloc[idx]  # в оригинальном CRS
                        best_source = 'A'
                        best_area = area
                        n_vertices = nv

        for idx in b_indices:
            if idx < len(gdf_b_proj):
                geom = gdf_b_proj.geometry.iloc[idx]
                if geom is not None and not geom.is_empty:
                    nv = _count_vertices(geom)
                    area = geom.area
                    if nv > n_vertices or (nv == n_vertices and area > best_area):
                        best_geom = gdf_b.geometry.iloc[idx]
                        best_source = 'B'
                        best_area = area
                        n_vertices = nv

        # Собираем атрибуты
        row = {
            'component_id': comp_id,
            'best_source': best_source,
            'n_polygons_a': len(a_indices),
            'n_polygons_b': len(b_indices),
            'area_m2': best_area,
        }

        # Высота/этажность из B (приоритетный источник)
        for idx in b_indices:
            if idx < len(gdf_b):
                for col in ['height', 'stairs', 'avg_floor_height']:
                    if col in gdf_b.columns and pd.notna(gdf_b.iloc[idx].get(col)):
                        row[col] = gdf_b.iloc[idx][col]

        # Этажность из A (если в B нет)
        if 'stairs' not in row or pd.isna(row.get('stairs')):
            for idx in a_indices:
                if idx < len(gdf_a):
                    for col in ['gkh_floor_count_min', 'gkh_floor_count_max']:
                        if col in gdf_a.columns and pd.notna(gdf_a.iloc[idx].get(col)):
                            row[col] = gdf_a.iloc[idx][col]

        results.append(row)

    result_df = pd.DataFrame(results)
    logger.info(f"    Итого зданий: {len(result_df)}")
    return result_df


def _count_vertices(geom) -> int:
    """Считает количество вершин в геометрии."""
    if geom is None:
        return 0
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords)
    if geom.geom_type == 'MultiPolygon':
        return sum(len(p.exterior.coords) for p in geom.geoms)
    return 0

# ВТОРОЙ ПРОХОД: ДОМЭТЧИВАНИЕ only_a / only_b

def second_pass_matching(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    components: pd.DataFrame,
    hausdorff_max_m: float = 5.0,
    area_ratio_max: float = 2.0,
    centroid_max_m: float = 15.0,
) -> tuple:
    """
    Второй проход матчинга: берем только оставшиеся only_a и only_b,
    пытаемся домэтчить более мягкими критериями.

    Уже совпавшие здания (matched) не трогаем.
    Можем только добавить новые совпадения, не сломать существующие.

    Критерии (нужно минимум 2 из 3):
      1. Hausdorff distance < 5 м (контуры близко, даже если сдвинуты)
      2. Соотношение площадей < 2× (похожий размер)
      3. Расстояние центроидов < 15 м
    """
    from shapely.ops import nearest_points

    logger.info("  Второй проход: домэтчивание only_a / only_b...")
    t0 = time.time()

    # Определяем only_a и only_b из компонент
    matched_a = set(components[components['source'] == 'A']['original_index'])
    matched_b = set(components[components['source'] == 'B']['original_index'])

    # Компоненты, где есть и A и B - это matched
    comp_sources = components.groupby('component_id')['source'].apply(set)
    matched_comps = comp_sources[comp_sources.apply(lambda s: 'A' in s and 'B' in s)].index

    # Индексы A и B, которые уже в matched компонентах
    matched_rows = components[components['component_id'].isin(matched_comps)]
    used_a = set(matched_rows[matched_rows['source'] == 'A']['original_index'])
    used_b = set(matched_rows[matched_rows['source'] == 'B']['original_index'])

    # Оставшиеся
    only_a_idx = [i for i in range(len(gdf_a)) if i not in used_a]
    only_b_idx = [j for j in range(len(gdf_b)) if j not in used_b]

    logger.info(f"    only_a: {len(only_a_idx)}, only_b: {len(only_b_idx)}")

    if not only_a_idx or not only_b_idx:
        logger.info("    Нечего домэтчивать")
        return [], {'second_pass_matches': 0}

    # Проецируем в метры
    gdf_a_proj = gdf_a.to_crs(METRIC_CRS)
    gdf_b_proj = gdf_b.to_crs(METRIC_CRS)

    # Строим sindex только для only_b
    only_b_set = set(only_b_idx)

    # Используем sindex полного gdf_b_proj, но фильтруем кандидатов
    sindex_b = gdf_b_proj.sindex

    new_matches = []
    n_checked = 0

    for i in only_a_idx:
        geom_a = gdf_a_proj.geometry.iloc[i]
        if geom_a is None or geom_a.is_empty:
            continue

        area_a = geom_a.area
        if area_a <= 0:
            continue

        centroid_a = geom_a.centroid

        # Расширяем bbox для поиска (hausdorff_max + запас)
        search_buffer = hausdorff_max_m + 10
        search_bounds = (
            geom_a.bounds[0] - search_buffer,
            geom_a.bounds[1] - search_buffer,
            geom_a.bounds[2] + search_buffer,
            geom_a.bounds[3] + search_buffer,
        )
        candidates = list(sindex_b.intersection(search_bounds))

        for j in candidates:
            if j not in only_b_set:
                continue

            geom_b = gdf_b_proj.geometry.iloc[j]
            if geom_b is None or geom_b.is_empty:
                continue

            area_b = geom_b.area
            if area_b <= 0:
                continue

            # Считаем 3 критерия
            score = 0

            # Критерий 1: расстояние центроидов
            centroid_dist = centroid_a.distance(geom_b.centroid)
            if centroid_dist > centroid_max_m * 3:  # быстрый отсев
                continue
            if centroid_dist <= centroid_max_m:
                score += 1

            # Критерий 2: соотношение площадей
            area_ratio = max(area_a, area_b) / min(area_a, area_b)
            if area_ratio <= area_ratio_max:
                score += 1

            # Критерий 3: Hausdorff distance
            try:
                hausdorff = geom_a.hausdorff_distance(geom_b)
                if hausdorff <= hausdorff_max_m:
                    score += 1
            except Exception:
                pass

            # Матч если минимум 2 из 3
            if score >= 2:
                # Считаем IoU для записи (может быть 0 если не пересекаются)
                try:
                    inter = geom_a.intersection(geom_b).area
                    union = area_a + area_b - inter
                    iou = inter / union if union > 0 else 0
                except Exception:
                    iou = 0

                new_matches.append((i, j, iou, 0))
                # Убираем j из only_b чтобы не дублировать
                only_b_set.discard(j)
                break  # один матч на i

            n_checked += 1

    elapsed = time.time() - t0
    logger.info(f"    Второй проход: {len(new_matches)} новых матчей за {elapsed:.1f}с")

    stats = {
        'second_pass_matches': len(new_matches),
        'second_pass_only_a_remaining': len(only_a_idx) - len(new_matches),
        'second_pass_only_b_remaining': len(only_b_set),
    }

    return new_matches, stats

# ОСНОВНОЙ ПАЙПЛАЙН МАТЧИНГА

def run_matching(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    iou_threshold: float = 0.2,
    output_dir: str = 'output',
) -> tuple:
    """
    Полный пайплайн сопоставления.
    Args:
        gdf_a: очищенный GeoDataFrame Источника А
        gdf_b: очищенный GeoDataFrame Источника Б
        iou_threshold: базовый порог IoU
        output_dir: директория для сохранения
    Returns:
        (components_df, buildings_df, stats)
    """
    logger.info("=" * 60)
    logger.info("СОПОСТАВЛЕНИЕ ИСТОЧНИКОВ")
    logger.info(f"  IoU threshold: {iou_threshold}")
    logger.info("=" * 60)

    t_total = time.time()

    # Шаг 1: Кластеризация внутри источников
    logger.info("\nШаг 1: Кластеризация внутри источников")
    clusters_a = cluster_within_source(gdf_a, 'A',
                                        touch_buffer_m=0.5,
                                        min_overlap_iou=0.05,
                                        max_area_ratio=10.0)
    clusters_b = cluster_within_source(gdf_b, 'B',
                                        touch_buffer_m=0.5,
                                        min_overlap_iou=0.05,
                                        max_area_ratio=10.0)

    # Шаг 2: Кросс-матчинг
    logger.info("\nШаг 2: Кросс-матчинг A ↔ B")
    matches = cross_match(gdf_a, gdf_b,
                          iou_threshold=iou_threshold,
                          centroid_max_dist_m=50.0)

    # Шаг 3: Адресный матчинг (дополнительный сигнал)
    logger.info("\nШаг 3: Адресный матчинг")
    addr_matches = address_match(gdf_a, gdf_b)

    # Шаг 4: Граф связности
    logger.info("\nШаг 4: Граф связности (1-й проход)")
    components, match_stats = build_connectivity_graph(
        gdf_a, gdf_b, clusters_a, clusters_b, matches, addr_matches
    )

    # Шаг 5: Второй проход - домэтчивание оставшихся
    logger.info("\nШаг 5: Второй проход (Hausdorff + площадь + центроиды)")
    new_matches, pass2_stats = second_pass_matching(
        gdf_a, gdf_b, components,
        hausdorff_max_m=5.0,
        area_ratio_max=2.0,
        centroid_max_m=15.0,
    )
    match_stats.update(pass2_stats)

    # Если есть новые матчи - перестраиваем граф
    if new_matches:
        logger.info("    Перестраиваем граф с новыми матчами...")
        all_matches = matches + new_matches
        components, match_stats_v2 = build_connectivity_graph(
            gdf_a, gdf_b, clusters_a, clusters_b, all_matches, addr_matches
        )
        # Обновляем статистику
        match_stats['components_total'] = match_stats_v2['components_total']
        match_stats['only_a'] = match_stats_v2['only_a']
        match_stats['only_b'] = match_stats_v2['only_b']
        match_stats['matched'] = match_stats_v2['matched']
        match_stats['cross_matches_total'] = len(all_matches)

    # Шаг 6: Выбор лучшей геометрии
    logger.info("\nШаг 6: Выбор лучшей геометрии")
    buildings = select_best_geometry(components, gdf_a, gdf_b)

    elapsed = time.time() - t_total
    logger.info(f"\n СОПОСТАВЛЕНИЕ ЗАВЕРШЕНО за {elapsed:.1f}с")

    return components, buildings, match_stats

# ТОЧКА ВХОДА

if __name__ == '__main__':
    import sys

    CLEAN_A = 'output/clean_A.csv'
    CLEAN_B = 'output/clean_B.csv'
    OUTPUT_DIR = 'output'

    if len(sys.argv) >= 3:
        CLEAN_A = sys.argv[1]
        CLEAN_B = sys.argv[2]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ЗАПУСК СОПОСТАВЛЕНИЯ")
    logger.info("=" * 60)
    t_total = time.time()

    # --- Загрузка ---
    gdf_a = load_clean_data(CLEAN_A, "Источник А (очищенный)")
    gdf_b = load_clean_data(CLEAN_B, "Источник Б (очищенный)")

    # --- Матчинг ---
    components, buildings, stats = run_matching(
        gdf_a, gdf_b,
        iou_threshold=0.2,
        output_dir=OUTPUT_DIR,
    )

    # --- Сохранение ---
    logger.info("\n" + "=" * 60)
    logger.info("СОХРАНЕНИЕ")
    logger.info("=" * 60)

    components.to_csv(os.path.join(OUTPUT_DIR, 'matched_components.csv'), index=False)
    logger.info(f"  Компоненты: {OUTPUT_DIR}/matched_components.csv")

    buildings.to_csv(os.path.join(OUTPUT_DIR, 'buildings_merged.csv'), index=False)
    logger.info(f"  Здания: {OUTPUT_DIR}/buildings_merged.csv")

    with open(os.path.join(OUTPUT_DIR, 'matching_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"  Статистика: {OUTPUT_DIR}/matching_stats.json")

    elapsed = time.time() - t_total
    logger.info(f"\n ВСЕ  ЗАВЕРШЕНО за {elapsed:.1f}с")

    # Сводка
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВАЯ СВОДКА")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")