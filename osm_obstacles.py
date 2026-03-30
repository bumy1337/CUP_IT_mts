"""
Скрипт: Загрузка НЕ-зданий из OSM, влияющих на распространение сигнала

КАКИЕ ОБЪЕКТЫ СКАЧИВАЕМ:

  - man_made=tower, mast, chimney, communications_tower (вышки, трубы, мачты)
  - historic=monument, memorial, column (памятники, колонны)
  - bridge=yes (мосты)
  - man_made=crane (краны)
  - power=tower (опоры ЛЭП)

ЗАЧЕМ:
Эти объекты НЕ являются зданиями, но физически блокируют / отражают сигнал.
Колонна 47 м посреди площади создает теневую зону.
Мост с вантами до 120 м - серьезное препятствие.
Труба ТЭЦ 100+ м - доминанта района.
"""

import pandas as pd
import numpy as np
import requests
import warnings
import logging
import time
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPB_BBOX = {
    'south': 59.75, 'west': 29.40,
    'north': 60.15, 'east': 30.75,
}


def download_obstacles(bbox=SPB_BBOX, cache_file='output/osm_obstacles_cache.json'):
    """
    Скачивает не-здания из OSM, которые влияют на сигнал.
    """
    if os.path.exists(cache_file):
        logger.info(f"  Загрузка из кэша: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    logger.info("  Скачивание объектов-препятствий из OSM...")

    bb = f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"

    query = f"""
    [out:json][timeout:300];
    (
      // Вышки, мачты, трубы
      node["man_made"+-"tower|mast|chimney|communications_tower"]({bb});
      way["man_made"+-"tower|mast|chimney|communications_tower"]({bb});

      // Памятники, колонны
      node["historic"+-"monument|memorial|column"]({bb});
      way["historic"+-"monument|memorial|column"]({bb});

      // Мосты
      way["bridge"="yes"]["layer"+-"1|2|3"]({bb});

      // Опоры ЛЭП
      node["power"="tower"]({bb});

      // Краны (строительные)
      node["man_made"="crane"]({bb});
    );
    out body;
    >;
    out skel qt;
    """

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for url in servers:
        logger.info(f"  Попытка: {url}")
        try:
            t0 = time.time()
            resp = requests.post(url, data={'data': query}, timeout=300,
                                  headers={'User-Agent': 'MTS-CupIT-2026/1.0'})
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"  Получено за {time.time()-t0:.1f}с")

            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            return data
        except Exception as e:
            logger.warning(f"  Ошибка: {e}")
            time.sleep(5)

    logger.error("  Все серверы недоступны")
    return None


def parse_obstacles(osm_data):
    """
    Парсит объекты из OSM JSON.
    Извлекает: тип, название, высоту (если есть), координаты.
    """
    if osm_data is None:
        return pd.DataFrame()

    elements = osm_data.get('elements', [])

    # Собираем nodes для координат
    nodes = {}
    for el in elements:
        if el['type'] == 'node' and 'lat' in el:
            nodes[el['id']] = (el['lat'], el['lon'])

    obstacles = []

    for el in elements:
        tags = el.get('tags', {})
        if not tags:
            continue

        # Определяем тип объекта
        obj_type = None
        obj_category = None

        if tags.get('man_made') in ('tower', 'mast', 'chimney', 'communications_tower', 'crane'):
            obj_type = tags['man_made']
            obj_category = 'infrastructure'
        elif tags.get('historic') in ('monument', 'memorial', 'column'):
            obj_type = tags['historic']
            obj_category = 'monument'
        elif tags.get('bridge') == 'yes':
            obj_type = 'bridge'
            obj_category = 'bridge'
        elif tags.get('power') == 'tower':
            obj_type = 'power_tower'
            obj_category = 'infrastructure'
        else:
            continue

        # Координаты
        lat = lon = None
        if el['type'] == 'node':
            lat, lon = el.get('lat'), el.get('lon')
        elif el['type'] == 'way':
            # Берем центр way
            way_nodes = el.get('nodes', [])
            lats, lons = [], []
            for nid in way_nodes:
                if nid in nodes:
                    lats.append(nodes[nid][0])
                    lons.append(nodes[nid][1])
            if lats:
                lat, lon = np.mean(lats), np.mean(lons)

        if lat is None or lon is None:
            continue

        # Высота
        height = None
        for h_tag in ['height', 'ele', 'building:height']:
            if h_tag in tags:
                try:
                    h_str = tags[h_tag].replace('m', '').replace(' ', '').replace(',', '.')
                    height = float(h_str)
                    if height <= 0 or height > 1000:
                        height = None
                except:
                    pass

        # Оценка высоты по типу если нет данных
        if height is None:
            default_heights = {
                'tower': 30, 'communications_tower': 50, 'mast': 40,
                'chimney': 60, 'crane': 40, 'power_tower': 35,
                'monument': 15, 'memorial': 5, 'column': 20,
                'bridge': 15,
            }
            height = default_heights.get(obj_type)

        name = tags.get('name', tags.get('name:ru', f'{obj_type}'))

        obstacles.append({
            'osm_id': el['id'],
            'name': name,
            'type': obj_type,
            'category': obj_category,
            'height': height,
            'height_source': 'osm_tag' if height and h_tag in tags else 'estimated',
            'lat': lat,
            'lon': lon,
        })

    df = pd.DataFrame(obstacles)
    logger.info(f"  Распарсено объектов: {len(df)}")
    return df


def analyze_obstacles(df):
    """Анализирует найденные препятствия."""
    logger.info("\n" + "=" * 60)
    logger.info("АНАЛИЗ ПРЕПЯТСТВИЙ ДЛЯ СИГНАЛА")
    logger.info("=" * 60)

    if len(df) == 0:
        logger.info("  Нет данных")
        return {}

    stats = {'total': len(df)}

    # По категориям
    logger.info("\n  По категориям:")
    for cat, grp in df.groupby('category'):
        n = len(grp)
        avg_h = grp['height'].mean() if grp['height'].notna().any() else 0
        max_h = grp['height'].max() if grp['height'].notna().any() else 0
        logger.info(f"    {cat}: {n} объектов, средняя высота {avg_h:.0f} м, макс {max_h:.0f} м")
        stats[cat] = {'count': n, 'avg_height': round(avg_h, 1), 'max_height': round(max_h, 1)}

    # По типам
    logger.info("\n  По типам:")
    for typ, grp in df.groupby('type'):
        n = len(grp)
        logger.info(f"    {typ}: {n}")
        stats[f'type_{typ}'] = n

    # Самые высокие
    if df['height'].notna().any():
        top = df.nlargest(10, 'height')
        logger.info("\n  Топ-10 самых высоких препятствий:")
        for _, row in top.iterrows():
            logger.info(f"    {row['name']}: {row['height']:.0f} м ({row['type']})")

    # Рекомендации
    tall = df[df['height'] > 30]
    logger.info(f"\n  Препятствий выше 30 м: {len(tall)}")
    logger.info("  - Эти объекты создают значимые теневые зоны для сигнала")
    logger.info("  - Рекомендуется учитывать при размещении базовых станций")

    bridges = df[df['category'] == 'bridge']
    if len(bridges) > 0:
        logger.info(f"\n  Мосты: {len(bridges)}")
        logger.info("  - Мосты создают помехи для сигнала под ними")
        logger.info("  - На крупных мостах рекомендуются репитеры")

    stats['tall_obstacles_30m'] = len(tall)
    stats['bridges'] = len(bridges)

    return stats


if __name__ == '__main__':
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ЗАГРУЗКА ПРЕПЯТСТВИЙ ДЛЯ СИГНАЛА ИЗ OSM")
    logger.info("=" * 60)
    t0 = time.time()

    # Скачиваем
    osm_data = download_obstacles()
    if osm_data is None:
        logger.error("Не удалось скачать данные")
        exit(1)

    # Парсим
    obstacles = parse_obstacles(osm_data)

    # Анализируем
    stats = analyze_obstacles(obstacles)

    # Сохраняем
    obstacles.to_csv(os.path.join(OUTPUT_DIR, 'osm_obstacles.csv'), index=False)
    logger.info(f"\n  Данные: {OUTPUT_DIR}/osm_obstacles.csv")

    with open(os.path.join(OUTPUT_DIR, 'osm_obstacles_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n ЗАВЕРШЕНО за {time.time()-t0:.1f}с")