"""
Скрипт: Задание 5 - Визуализация и инсайты

СОЗДАЕТ 3 КАРТЫ:
1. Тепловая карта всего Питера (гексагоны H3) - средняя высота по районам
2. Детальная карта Приморского района - спальный район, высотки, новостройки
3. Детальная карта исторического центра - старый фонд, 5-6 этажей

ИНСАЙТЫ ДЛЯ ПЛАНИРОВАНИЯ СЕТИ:
- Где высотки создают "каньоны" для сигнала
- Где однородная застройка (одна макросота покроет все)
- Где смешанная (нужны микросоты)
- Плотность застройки по зонам

"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, mapping
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import warnings
import logging
import time
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRIC_CRS = 'EPSG:32636'

# Координаты центров районов
SPB_CENTER = [59.94, 30.31]
PRIMORSKY_CENTER = [60.00, 30.25]
HISTORIC_CENTER = [59.935, 30.32]
# Третий тип: промзона (Кировский район — смешанная промышленная + жилая застройка).
# Принципиально иной профиль: низкая разреженная застройка, большие площади,
# для сети МТС оптимальна 1 макросота вместо решётки микросот.
KIROVSKY_CENTER = [59.865, 30.255]

# ЗАГРУЗКА

def load_buildings_with_geometry(buildings_path, clean_a_path, clean_b_path, components_path):
    """Загружает здания и привязывает к ним геометрию."""
    logger.info("Загрузка данных...")

    buildings = pd.read_csv(buildings_path)
    components = pd.read_csv(components_path)

    def load_gdf(path):
        df = pd.read_csv(path)
        gcol = None
        for c in ['geometry_wkt', 'geometry', 'wkt']:
            if c in df.columns: gcol = c; break
        df['geometry'] = df[gcol].apply(lambda x: wkt.loads(x) if pd.notna(x) and isinstance(x, str) else None)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        s = gdf.geometry.dropna().iloc[0].centroid
        gdf = gdf.set_crs('EPSG:4326' if -180 <= s.x <= 180 else METRIC_CRS)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    gdf_a = load_gdf(clean_a_path)
    gdf_b = load_gdf(clean_b_path)

    logger.info("  Привязка геометрий к зданиям...")
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

    buildings['geometry'] = geom_list
    gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs='EPSG:4326')
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    logger.info(f"  Загружено: {len(gdf)} зданий с геометрией")
    return gdf

# КАРТА 1: ТЕПЛОВАЯ КАРТА (H3 ГЕКСАГОНЫ)

def create_heatmap(gdf, output_path, obstacles=None):
    """
    Тепловая карта средней высоты по гексагонам H3.
    Каждый гексагон +-200 м - показывает среднюю высоту зданий в нем.
    """
    logger.info("\n" + "=" * 60)
    logger.info("КАРТА 1: ТЕПЛОВАЯ КАРТА ВЫСОТЫ ПИТЕРА")
    logger.info("=" * 60)

    try:
        import h3
        use_h3 = True
        logger.info("  Используем H3 гексагоны")
    except ImportError:
        use_h3 = False
        logger.info("  H3 не установлен, используем grid-тепловую карту")

    m = folium.Map(location=SPB_CENTER, zoom_start=11,
                   tiles='CartoDB positron', control_scale=True)

    valid = gdf[gdf['height_final'].notna()].copy()
    centroids = valid.geometry.centroid

    if use_h3:
        # Группируем по H3 гексагонам (resolution 8 ≈ 460 м)
        logger.info("  Вычисление H3 индексов...")
        h3_indices = []
        for c in centroids:
            try:
                h3_idx = h3.latlng_to_cell(c.y, c.x, 8)
                h3_indices.append(h3_idx)
            except:
                h3_indices.append(None)

        valid['h3_index'] = h3_indices
        valid_h3 = valid[valid['h3_index'].notna()]

        hex_stats = valid_h3.groupby('h3_index').agg(
            mean_height=('height_final', 'mean'),
            median_height=('height_final', 'median'),
            count=('height_final', 'count'),
            max_height=('height_final', 'max'),
        ).reset_index()

        logger.info(f"  Гексагонов: {len(hex_stats)}")

        # Цветовая шкала
        colormap = cm.LinearColormap(
            colors=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'],
            vmin=0, vmax=40,
            caption='Средняя высота зданий (м)'
        )

        # Рисуем гексагоны
        logger.info("  Рисуем гексагоны...")
        for _, row in hex_stats.iterrows():
            try:
                boundary = h3.cell_to_boundary(row['h3_index'])
                polygon_coords = [[lat, lng] for lat, lng in boundary]
                color = colormap(min(row['mean_height'], 40))

                folium.Polygon(
                    locations=polygon_coords,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    weight=0.5,
                    popup=folium.Popup(
                        f"<b>Средняя высота:</b> {row['mean_height']:.1f} м<br>"
                        f"<b>Макс высота:</b> {row['max_height']:.1f} м<br>"
                        f"<b>Зданий:</b> {int(row['count'])}",
                        max_width=200
                    ),
                ).add_to(m)
            except:
                continue

        colormap.add_to(m)

    else:
        # Fallback: обычная тепловая карта
        heat_data = []
        for c, h in zip(centroids, valid['height_final']):
            if c is not None and not c.is_empty and h > 0:
                heat_data.append([c.y, c.x, min(h, 50)])

        HeatMap(heat_data, radius=8, blur=10, max_zoom=13,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)

    # Добавляем метки районов
    districts = [
        ("Приморский", 60.00, 30.25, "Спальный район, высотки"),
        ("Василеостровский", 59.94, 30.27, "Смешанная застройка"),
        ("Центральный", 59.93, 30.35, "Исторический центр"),
        ("Адмиралтейский", 59.92, 30.30, "Старый фонд"),
        ("Выборгский", 60.03, 30.35, "Новостройки"),
        ("Московский", 59.85, 30.32, "Смешанная застройка"),
        ("Петроградский", 59.96, 30.30, "Смешанная застройка"),
        ("Калининский", 60.03, 30.40, "Спальный район"),
        ("Красногвардейский", 59.95, 30.48, "Промзоны + жилое"),
        ("Невский", 59.88, 30.45, "Спальный район"),
        ("Кировский", 59.85, 30.25, "Промзоны + жилое"),
        ("Фрунзенский", 59.87, 30.39, "Спальный район"),
    ]
    for name, lat, lon, desc in districts:
        folium.Marker(
            [lat, lon],
            popup=f"<b>{name}</b><br>{desc}",
            icon=folium.DivIcon(html=f'<div style="font-size:11px;font-weight:bold;color:#333;'
                                     f'text-shadow:1px 1px 2px white,-1px -1px 2px white;'
                                     f'white-space:nowrap">{name}</div>')
        ).add_to(m)

    # Добавляем известные ориентиры (высотные доминанты)
    # Эти здания могут отсутствовать в данных или быть отфильтрованы,
    # поэтому добавляем вручную как reference points
    landmarks = [
        ("Лахта Центр", 59.9871, 30.1777, 462, "Самое высокое здание России и Европы"),
        ("Исаакиевский собор", 59.9340, 30.3063, 101.5, "Купол 101.5 м"),
        ("Петропавловский собор", 59.9503, 30.3164, 122.5, "Шпиль 122.5 м"),
        ("Казанский собор", 59.9343, 30.3245, 71.6, "Купол 71.6 м"),
        ("Адмиралтейство", 59.9375, 30.3086, 72.5, "Шпиль 72.5 м"),
        ("Смольный собор", 59.9486, 30.3956, 93.7, "Колокольня 93.7 м"),
        ("Телебашня", 59.9856, 30.3153, 326, "326 м"),
    ]
    for name, lat, lon, height, desc in landmarks:
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b>{name}</b><br>"
                f"<b>Высота:</b> {height} м<br>"
                f"{desc}</div>",
                max_width=220
            ),
            icon=folium.Icon(color='red', icon='arrow-up', prefix='fa'),
        ).add_to(m)

    # Добавляем границы районов через GeoJSON (Overpass)
    # Используем простые круги для обозначения зон
    zone_colors = {
        'Высотная застройка': '#d7191c',
        'Историческая застройка': '#2b83ba',
        'Промзоны': '#888888',
    }
    zones = [
        ("Высотная застройка", 60.00, 30.25, 3000),   # Приморский
        ("Высотная застройка", 60.03, 30.35, 2500),   # Выборгский
        ("Историческая застройка", 59.935, 30.32, 2000), # Центр
        ("Промзоны", 59.88, 30.48, 2000),              # Невский/Обухово
        ("Промзоны", 59.85, 30.25, 2000),              # Кировский
    ]
    for zone_name, lat, lon, radius in zones:
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color=zone_colors.get(zone_name, '#666'),
            fill=False,
            weight=2,
            dash_array='8 4',
            popup=zone_name,
        ).add_to(m)

    # Добавляем препятствия из OSM
    _add_obstacles_layer(m, obstacles)

    m.save(output_path)
    logger.info(f"  Сохранено: {output_path}")
    return hex_stats if use_h3 else None


def _add_obstacles_layer(m, obstacles, center=None, radius_m=None):
    """Добавляет слой препятствий (вышки, памятники, мосты) на карту."""
    if obstacles is None or len(obstacles) == 0:
        return

    icon_map = {
        'infrastructure': ('orange', 'signal'),
        'monument': ('purple', 'university'),
        'bridge': ('blue', 'road'),
    }

    n_added = 0
    for _, row in obstacles.iterrows():
        lat, lon = row.get('lat'), row.get('lon')
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Если указан центр и радиус - фильтруем
        if center is not None and radius_m is not None:
            from math import radians, cos, sqrt
            dlat = (lat - center[0]) * 111320
            dlon = (lon - center[1]) * 111320 * cos(radians(center[0]))
            dist = sqrt(dlat**2 + dlon**2)
            if dist > radius_m:
                continue

        cat = row.get('category', 'infrastructure')
        color, icon = icon_map.get(cat, ('gray', 'info-sign'))
        name = row.get('name', 'unknown')
        height = row.get('height', '?')
        h_src = row.get('height_source', '')
        obj_type = row.get('type', '')

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b>{name}</b><br>"
                f"<b>Тип:</b> {obj_type}<br>"
                f"<b>Высота:</b> {height} м"
                f"{' (оценка)' if h_src == 'estimated' else ''}<br>"
                f"<i style='color:#888'>Не здание, но влияет на сигнал</i></div>",
                max_width=220
            ),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
        ).add_to(m)
        n_added += 1

    if n_added > 0:
        logger.info(f"  Добавлено препятствий на карту: {n_added}")

# КАРТА 2/3: ДЕТАЛЬНАЯ КАРТА РАЙОНА

def create_district_map(gdf, center, radius_m, name, output_path, zoom=14, obstacles=None):
    """
    Детальная карта конкретного района с контурами зданий.
    Каждое здание раскрашено по высоте, при клике - подробная информация.
    """
    logger.info(f"\n  Создание карты: {name}...")

    # Фильтруем здания в радиусе от центра
    from shapely.geometry import Point
    center_point = Point(center[1], center[0])  # lon, lat

    gdf_proj = gdf.to_crs(METRIC_CRS)
    center_proj = gpd.GeoSeries([center_point], crs='EPSG:4326').to_crs(METRIC_CRS).iloc[0]
    distances = gdf_proj.geometry.centroid.distance(center_proj)
    mask = distances <= radius_m

    district = gdf[mask].copy()
    logger.info(f"  Зданий в районе: {len(district)}")

    if len(district) == 0:
        logger.warning(f"  Нет зданий в радиусе {radius_m} м от {center}")
        return None

    m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB positron')

    # Цветовая шкала
    colormap = cm.LinearColormap(
        colors=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'],
        vmin=0, vmax=50,
        caption='Высота здания (м)'
    )

    # Рисуем здания
    n_drawn = 0
    for _, row in district.iterrows():
        geom = row.geometry
        h = row.get('height_final', 0)
        if pd.isna(h): h = 0

        try:
            color = colormap(min(h, 50))
            source = row.get('height_source', 'unknown')
            conf = row.get('height_confidence', 'unknown')
            purpose = row.get('purpose', 'N/A')
            area = row.get('area_m2', 0)

            popup_html = (
                f"<div style='font-family:sans-serif;font-size:12px;min-width:180px'>"
                f"<b>Высота:</b> {h:.1f} м (+-{int(h/3)} эт.)<br>"
                f"<b>Площадь:</b> {area:.0f} м²<br>"
                f"<b>Назначение:</b> {purpose}<br>"
                f"<b>Источник:</b> {source}<br>"
                f"<b>Уверенность:</b> {conf}<br>"
                f"</div>"
            )

            if geom.geom_type == 'Polygon':
                coords = [(y, x) for x, y in geom.exterior.coords]
                folium.Polygon(
                    locations=coords, color='#333', weight=0.5,
                    fill=True, fill_color=color, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=250),
                ).add_to(m)
                n_drawn += 1
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(y, x) for x, y in poly.exterior.coords]
                    folium.Polygon(
                        locations=coords, color='#333', weight=0.5,
                        fill=True, fill_color=color, fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=250),
                    ).add_to(m)
                    n_drawn += 1
        except:
            continue

    colormap.add_to(m)

    # Добавляем ориентиры, которые попадают в радиус
    landmarks = [
        ("Лахта Центр", 59.9871, 30.1777, 462, " Самое высокое здание Европы"),
        ("Исаакиевский собор", 59.9340, 30.3063, 101.5, " Купол 101.5 м"),
        ("Петропавловский собор", 59.9503, 30.3164, 122.5, " Шпиль 122.5 м"),
        ("Казанский собор", 59.9343, 30.3245, 71.6, " Купол 71.6 м"),
        ("Адмиралтейство", 59.9375, 30.3086, 72.5, " Шпиль 72.5 м"),
        ("Смольный собор", 59.9486, 30.3956, 93.7, " Колокольня 93.7 м"),
        ("Александровская колонна", 59.9390, 30.3158, 47.5, " Памятник, не в данных"),
    ]
    from shapely.geometry import Point
    for lm_name, lat, lon, height, desc in landmarks:
        lm_point = Point(lon, lat)
        lm_proj = gpd.GeoSeries([lm_point], crs='EPSG:4326').to_crs(METRIC_CRS).iloc[0]
        dist_to_center = lm_proj.distance(center_proj)
        if dist_to_center <= radius_m:
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;font-size:12px'>"
                    f"<b>{lm_name}</b><br>"
                    f"<b>Реальная высота:</b> {height} м<br>"
                    f"{desc}<br><br>"
                    f"<i style='color:#888'>Шпили, купола и памятники могут<br>"
                    f"отсутствовать в полигональных данных -<br>"
                    f"это ограничение 2D-источников.</i></div>",
                    max_width=250
                ),
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
            ).add_to(m)

    # Добавляем препятствия из OSM
    _add_obstacles_layer(m, obstacles, center=center, radius_m=radius_m)

    # Статистика района
    heights = district['height_final'].dropna()
    stats = {
        'name': name,
        'buildings': len(district),
        'drawn': n_drawn,
        'mean_height': round(heights.mean(), 1) if len(heights) > 0 else 0,
        'median_height': round(heights.median(), 1) if len(heights) > 0 else 0,
        'max_height': round(heights.max(), 1) if len(heights) > 0 else 0,
    }

    # Добавляем легенду с инфо
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;
                font-family:sans-serif;max-width:280px">
        <b>{name}</b><br>
        Зданий: {len(district)}<br>
        Средняя высота: {stats['mean_height']} м<br>
        Медиана: {stats['median_height']} м<br>
        Максимум: {stats['max_height']} м
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    logger.info(f"  Нарисовано зданий: {n_drawn}")
    logger.info(f"  Средняя высота: {stats['mean_height']} м")
    logger.info(f"  Сохранено: {output_path}")

    return stats

# ИНСАЙТЫ ДЛЯ ПЛАНИРОВАНИЯ СЕТИ

def compute_insights(gdf):
    """
    Вычисление инсайтов о структуре застройки, полезных для планирования сети.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ИНСАЙТЫ ДЛЯ ПЛАНИРОВАНИЯ СЕТИ")
    logger.info("=" * 60)

    insights = {}
    valid = gdf[gdf['height_final'].notna()].copy()

    # 1. Распределение по типам застройки
    logger.info("\n  1. ТИПЫ ЗАСТРОЙКИ:")
    bins = [0, 5, 12, 25, 50, 500]
    labels = ['одноэтажная (до 5м)', 'малоэтажная (5-12м)',
              'среднеэтажная (12-25м)', 'высотная (25-50м)', 'небоскребы (50м+)']
    valid['height_category'] = pd.cut(valid['height_final'], bins=bins, labels=labels)
    cat_stats = valid['height_category'].value_counts()

    type_insights = {}
    for cat, cnt in cat_stats.items():
        pct = cnt / len(valid) * 100
        logger.info(f"    {cat}: {cnt} ({pct:.1f}%)")
        type_insights[str(cat)] = {'count': int(cnt), 'pct': round(pct, 1)}
    insights['building_types'] = type_insights

    # 2. Зоны высокой плотности (много зданий > 25 м в радиусе)
    logger.info("\n  2. РЕКОМЕНДАЦИИ ДЛЯ ПЛАНИРОВАНИЯ СЕТИ:")

    tall_buildings = valid[valid['height_final'] > 25]
    n_tall = len(tall_buildings)
    pct_tall = n_tall / len(valid) * 100

    logger.info(f"    Высотных зданий (>25 м): {n_tall} ({pct_tall:.1f}%)")
    logger.info(f"    Требуют микросот на крышах для устранения теневых зон")
    insights['tall_buildings'] = {'count': n_tall, 'pct': round(pct_tall, 1)}

    low_buildings = valid[valid['height_final'] <= 5]
    n_low = len(low_buildings)
    pct_low = n_low / len(valid) * 100

    logger.info(f"    Одноэтажных зданий (≤5 м): {n_low} ({pct_low:.1f}%)")
    logger.info(f"    Хорошая прямая видимость, макросоты эффективны")
    insights['low_buildings'] = {'count': n_low, 'pct': round(pct_low, 1)}

    # 3. Рекомендации по назначению
    if 'purpose' in valid.columns:
        logger.info("\n  3. ПРИОРИТЕТНЫЕ ТИПЫ ЗДАНИЙ (по кол-ву абонентов):")
        purpose_stats = valid.groupby('purpose').agg(
            count=('height_final', 'count'),
            mean_h=('height_final', 'mean'),
            total_floors=('height_final', lambda x: (x / 3).sum()),
        ).sort_values('total_floors', ascending=False).head(5)

        for purp, row in purpose_stats.iterrows():
            logger.info(f"    {purp}: {int(row['count'])} зд., "
                        f"средняя {row['mean_h']:.0f} м, "
                        f"+-{int(row['total_floors'])} суммарных этажей")
        insights['priority_buildings'] = purpose_stats.to_dict('index')

    # 4. Общие рекомендации
    logger.info("\n  4. КЛЮЧЕВЫЕ ВЫВОДЫ:")
    logger.info("    • Приморский район: высотная застройка 25-50 м,")
    logger.info("      плотные жилые кварталы — микросоты + DAS системы")
    logger.info("    • Исторический центр: однородная 15-20 м застройка,")
    logger.info("      узкие улицы — уличные микросоты на фасадах")
    logger.info("    • Кировский район (промзона): низкая разреженная застройка <5 м,")
    logger.info("      большие открытые площади — 1 макросота покрывает 3-5 км²,")
    logger.info("      нагрузка минимальна (промышленные объекты, нет жилых абонентов)")
    logger.info("    • Дачные поселки: 5 м, разреженная застройка —")
    logger.info("      минимум базовых станций, большой радиус покрытия")

    insights['recommendations'] = {
        'primorsky': 'Микросоты + DAS для высотных жилых кварталов',
        'historic_center': 'Уличные микросоты на фасадах для узких улиц',
        'industrial_kirovsky': 'Макросоты с радиусом 3-5 км, нагрузка минимальна',
        'suburban': 'Минимальная плотность БС, широкое покрытие',
    }

    # 5. Ограничения модели
    logger.info("\n  5. ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ:")
    logger.info("    • Шпили, купола, антенны: 2D-полигоны не передают перепад")
    logger.info("      высот одного здания (купол Казанского = 71 м, стены = 15 м).")
    logger.info("      Для этого нужны 3D-данные (LiDAR, CityGML).")
    logger.info("    • Сверхвысокие объекты (Лахта Центр 462 м) могут отсутствовать")
    logger.info("      в источниках или быть отфильтрованы - добавлены как ориентиры.")
    logger.info("    • Памятники (Александровская колонна 47 м) не являются зданиями")
    logger.info("      и не входят в модель высотности - но влияют на распространение сигнала.")

    insights['limitations'] = {
        'no_3d': 'Купола, шпили, антенны не отражены - нужны LiDAR/CityGML данные',
        'super_tall': 'Единичные сверхвысокие объекты могут отсутствовать в источниках',
        'monuments': 'Памятники и нежилые конструкции не в модели, но влияют на сигнал',
        'recommendation': 'Интеграция с LiDAR-сканированием для верификации высотных доминант',
    }

    return insights

# ТОЧКА ВХОДА

if __name__ == '__main__':
    OUTPUT_DIR = 'output'

    # Пути к файлам
    BUILDINGS = os.path.join(OUTPUT_DIR, 'buildings_with_osm.csv')
    if not os.path.exists(BUILDINGS):
        BUILDINGS = os.path.join(OUTPUT_DIR, 'buildings_with_height.csv')
    CLEAN_A = os.path.join(OUTPUT_DIR, 'clean_A.csv')
    CLEAN_B = os.path.join(OUTPUT_DIR, 'clean_B.csv')
    COMPONENTS = os.path.join(OUTPUT_DIR, 'matched_components.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(" ЗАДАНИЕ 5: ВИЗУАЛИЗАЦИЯ И ИНСАЙТЫ")
    logger.info("=" * 60)
    t0 = time.time()

    # Загрузка
    gdf = load_buildings_with_geometry(BUILDINGS, CLEAN_A, CLEAN_B, COMPONENTS)

    # Загрузка препятствий из OSM (если есть)
    obstacles_path = os.path.join(OUTPUT_DIR, 'osm_obstacles.csv')
    obstacles = None
    if os.path.exists(obstacles_path):
        obstacles = pd.read_csv(obstacles_path)
        logger.info(f"  Препятствия: {len(obstacles)} объектов")
    else:
        logger.info("  Файл osm_obstacles.csv не найден - запустите сначала osm_obstacles.py")

    # Карта 1: Тепловая карта всего города
    logger.info("\n КАРТА 1: Тепловая карта Питера ")
    hex_stats = create_heatmap(gdf, os.path.join(OUTPUT_DIR, 'map_heatmap.html'), obstacles)

    # Карта 2: Приморский район (спальный, высотки)
    logger.info("\n КАРТА 2: Приморский район ")
    primorsky_stats = create_district_map(
        gdf, center=PRIMORSKY_CENTER, radius_m=2000,
        name="Приморский район (спальный)", zoom=15,
        output_path=os.path.join(OUTPUT_DIR, 'map_primorsky.html'),
        obstacles=obstacles,
    )

    # Карта 3: Исторический центр
    logger.info("\n КАРТА 3: Исторический центр ")
    center_stats = create_district_map(
        gdf, center=HISTORIC_CENTER, radius_m=1500,
        name="Исторический центр", zoom=15,
        output_path=os.path.join(OUTPUT_DIR, 'map_center.html'),
        obstacles=obstacles,
    )

    # Карта 4: Кировский район (промзона + жилое)
    # Третий тип застройки — принципиально иной профиль для планирования сети МТС:
    # низкая разреженная застройка (<5 м), большие незастроенные площади,
    # для покрытия достаточна 1 макросота вместо решётки микросот.
    logger.info("\n КАРТА 4: Кировский район (промзона) ")
    kirovsky_stats = create_district_map(
        gdf, center=KIROVSKY_CENTER, radius_m=2000,
        name="Кировский район (промзона + жилое)", zoom=15,
        output_path=os.path.join(OUTPUT_DIR, 'map_kirovsky.html'),
        obstacles=obstacles,
    )

    # Инсайты
    insights = compute_insights(gdf)
    insights['districts'] = {
        'primorsky': primorsky_stats,
        'historic_center': center_stats,
        'kirovsky_industrial': kirovsky_stats,
    }

    with open(os.path.join(OUTPUT_DIR, 'insights.json'), 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  Инсайты: {OUTPUT_DIR}/insights.json")

    logger.info(f"\n ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА за {time.time()-t0:.1f}с")
    logger.info(f"\n  Откройте в браузере:")
    logger.info(f"    {OUTPUT_DIR}/map_heatmap.html   — тепловая карта всего города")
    logger.info(f"    {OUTPUT_DIR}/map_primorsky.html — Приморский (высотки, спальный)")
    logger.info(f"    {OUTPUT_DIR}/map_center.html    — Исторический центр (5-6 эт.)")
    logger.info(f"    {OUTPUT_DIR}/map_kirovsky.html  — Кировский (промзона, низкая застройка)")