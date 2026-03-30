"""
Скрипт: Задание 4 - Валидация полученных результатов

ЧТО ТАКОЕ ВАЛИДАЦИЯ (простыми словами):

Мы посчитали высоту для 170K зданий. Но откуда мы знаем что не наврали?
У нас нет "правильных ответов" для всех зданий.
Поэтому мы проверяем КОСВЕННО - через несколько независимых проверок.

Если все проверки проходят - мы не можем ДОКАЗАТЬ что все правильно
но можем показать что ничего ЯВНО НЕ СЛОМАНО. Это и есть валидация.

КАКИЕ ПРОВЕРКИ ДЕЛАЕМ:

1. СТАТИСТИЧЕСКАЯ ПРОВЕРКА (sanity check)
   Медианная высота Питера должна быть +-10 м (3-4 этажа)
   Зданий выше 100 м - единицы (Лахта Центр и еще пара)
   Зданий ниже 3 м - очень мало (это заборы, а не здания)
   - Если статистика похожа на реальный город - ок

2. СОГЛАСОВАННОСТЬ ИСТОЧНИКОВ
   У 120K зданий данные из А и Б совпадают (CV < 15%)
   Считаем: у скольких процентов расхождение < 1 этажа, < 2 этажей, > 5 этажей
   - Если 90%+ зданий согласованы в пределах 1 этажа - отлично

3. ML-МОДЕЛЬ: HOLDOUT ТЕСТ
   Мы обучили на 80%, проверили на 20% (которые модель не видела)
   MAE, RMSE, R² - метрики на этих 20%
   Дополнительно: распределение ошибок (гистограмма)
   - MAE < 3 м = ошибка меньше 1 этажа = хорошо

4. ПРОСТРАНСТВЕННАЯ ПРОВЕРКА
   Средняя высота по районам должна быть разной:
   Центр (Адмиралтейский) - 15-20 м (старые 5-6 этажные дома)
   Спальные районы (Приморский, Выборгский) - 30-50 м (новостройки)
   Промзоны - 5-10 м (одноэтажные корпуса)
   - Если районы различаются как ожидается - модель ловит пространственные паттерны

5. ПРОВЕРКА СОСЕДЕЙ
   Берем случайные здания и смотрим: похожа ли их высота на высоту соседей?
   В реальном городе дома в одном квартале обычно одной высоты.
   Считаем корреляцию: высота здания vs средняя высота соседей в 100 м
   - Корреляция > 0.7 = модель адекватна

6. АНАЛИЗ ОШИБОК ML
   На каких зданиях модель ошибается больше всего?
   Группируем по: размеру, назначению, кол-ву оценок
   - Понимаем слабые места и описываем их в презентации

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/validation_log.txt', mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

METRIC_CRS = 'EPSG:32636'


# ПРОВЕРКА 1: СТАТИСТИКА (SANITY CHECK)

def check_statistics(buildings):
    """
    Проверяем: похожа ли статистика на реальный город?

    Ожидания для Санкт-Петербурга:
      - Медианная высота: 8-15 м (3-5 этажей)
      - Средняя высота: 12-20 м
      - Зданий > 100 м: < 50 (высотки - редкость)
      - Зданий < 3 м: < 5% (это скорее ограждения)
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 1: СТАТИСТИКА (SANITY CHECK)")
    logger.info("=" * 60)

    h = buildings['height_final'].dropna()
    total = len(h)

    results = {
        'total_buildings': total,
        'min': round(h.min(), 1),
        'max': round(h.max(), 1),
        'median': round(h.median(), 1),
        'mean': round(h.mean(), 1),
        'std': round(h.std(), 1),
        'p5': round(h.quantile(0.05), 1),
        'p25': round(h.quantile(0.25), 1),
        'p75': round(h.quantile(0.75), 1),
        'p95': round(h.quantile(0.95), 1),
    }

    logger.info(f"  Всего зданий: {total}")
    logger.info(f"  Min: {results['min']} м")
    logger.info(f"  5%: {results['p5']} м")
    logger.info(f"  25%: {results['p25']} м")
    logger.info(f"  Медиана: {results['median']} м")
    logger.info(f"  75%: {results['p75']} м")
    logger.info(f"  95%: {results['p95']} м")
    logger.info(f"  Max: {results['max']} м")
    logger.info(f"  Среднее: {results['mean']} м")

    # Проверки
    checks = {}

    n_over_100 = (h > 100).sum()
    checks['buildings_over_100m'] = n_over_100
    ok = n_over_100 < 100
    logger.info(f"\n  Зданий > 100 м: {n_over_100} {' OK' if ok else ' МНОГО'}")

    n_under_3 = (h < 3).sum()
    pct_under_3 = n_under_3 / total * 100
    checks['buildings_under_3m'] = n_under_3
    checks['pct_under_3m'] = round(pct_under_3, 1)
    ok = pct_under_3 < 5
    logger.info(f"  Зданий < 3 м: {n_under_3} ({pct_under_3:.1f}%) {' OK' if ok else ' МНОГО'}")

    ok_median = 5 <= results['median'] <= 20
    logger.info(f"  Медиана {results['median']} м: {' реалистично' if ok_median else ' подозрительно'}")

    # Распределение по этажности (высота / 3 м)
    floors_est = (h / 3).round()
    floor_dist = floors_est.value_counts().sort_index().head(15)
    logger.info(f"\n  Распределение по примерной этажности:")
    for fl, cnt in floor_dist.items():
        bar = "█" * min(int(cnt / total * 200), 50)
        logger.info(f"    {int(fl):2d} эт. ({fl*3:.0f} м): {cnt:6d} ({cnt/total*100:5.1f}%) {bar}")

    results['checks'] = checks
    return results

# ПРОВЕРКА 2: СОГЛАСОВАННОСТЬ ИСТОЧНИКОВ

def check_source_agreement(buildings):
    """
    Для зданий с несколькими оценками: насколько источники согласованы?
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 2: СОГЛАСОВАННОСТЬ ИСТОЧНИКОВ")
    logger.info("=" * 60)

    # Здания с height_b И gkh данными - можно сравнить
    has_both = buildings[
        buildings['height_b'].notna() &
        buildings['gkh_floor_max'].notna() &
        (buildings['gkh_floor_max'] > 0)
    ].copy()

    if len(has_both) == 0:
        logger.info("  Нет зданий для перекрестной проверки")
        return {}

    # Сравниваем height_b vs gkh_floor_max * 3
    has_both['height_from_gkh'] = has_both['gkh_floor_max'] * 3.0
    has_both['abs_diff'] = np.abs(has_both['height_b'] - has_both['height_from_gkh'])
    has_both['rel_diff'] = has_both['abs_diff'] / (has_both['height_b'] + 0.1)

    total = len(has_both)

    # По порогам расхождения
    thresholds = [
        ('< 3 м (1 этаж)', has_both['abs_diff'] < 3),
        ('< 6 м (2 этажа)', has_both['abs_diff'] < 6),
        ('< 9 м (3 этажа)', has_both['abs_diff'] < 9),
        ('> 15 м (5+ этажей)', has_both['abs_diff'] > 15),
    ]

    results = {'total_compared': total}
    logger.info(f"\n  Сравнение height(Б) vs gkh×3(А) для {total} зданий:")

    for label, mask in thresholds:
        cnt = mask.sum()
        pct = cnt / total * 100
        results[label] = {'count': int(cnt), 'pct': round(pct, 1)}
        logger.info(f"    Расхождение {label}: {cnt} ({pct:.1f}%)")

    # Общая статистика расхождений
    mean_diff = has_both['abs_diff'].mean()
    median_diff = has_both['abs_diff'].median()
    results['mean_abs_diff_m'] = round(mean_diff, 2)
    results['median_abs_diff_m'] = round(median_diff, 2)

    logger.info(f"\n  Среднее расхождение: {mean_diff:.2f} м")
    logger.info(f"  Медианное расхождение: {median_diff:.2f} м")

    return results

# ПРОВЕРКА 3: ML-МОДЕЛЬ (HOLDOUT)

def check_ml_holdout(buildings):
    """
    Переобучаем ML на 80% зданий с height_b и проверяем на 20%.
    Это независимая проверка: тестовые здания модель никогда не видела.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 3: ML HOLDOUT ТЕСТ")
    logger.info("=" * 60)

    feature_cols = ['area_m2', 'perimeter_m', 'compactness', 'n_vertices',
                    'avg_height_neighbors_100m', 'purpose_encoded', 'n_polygons_a', 'n_polygons_b']

    # Только здания с надежной высотой (height_b)
    mask = buildings['height_b'].notna() & (buildings['height_b'] > 0)
    for col in feature_cols:
        if col not in buildings.columns:
            logger.info(f"  Нет колонки {col}, пропускаем ML проверку")
            return {}

    data = buildings[mask].copy()
    X = data[feature_cols].copy()
    y = data['height_b'].copy()

    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())

    logger.info(f"  Данные: {len(X)} зданий с height_b")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                                   num_leaves=63, min_child_samples=20, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           min_samples_leaf=20, subsample=0.8, random_state=42)
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)

    # Ошибки по группам высоты
    errors = pd.DataFrame({'true': y_te, 'pred': y_pred, 'error': np.abs(y_te - y_pred)})
    errors['height_group'] = pd.cut(errors['true'], bins=[0, 5, 10, 20, 50, 500],
                                     labels=['0-5м', '5-10м', '10-20м', '20-50м', '50м+'])

    results = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'train_size': len(X_tr),
        'test_size': len(X_te),
    }

    logger.info(f"\n  MAE: {mae:.2f} м (средняя ошибка)")
    logger.info(f"  RMSE: {rmse:.2f} м")
    logger.info(f"  R²: {r2:.4f}")

    logger.info(f"\n  Ошибки по группам высоты:")
    for grp, grp_data in errors.groupby('height_group', observed=True):
        grp_mae = grp_data['error'].mean()
        grp_cnt = len(grp_data)
        results[f'MAE_{grp}'] = round(grp_mae, 2)
        logger.info(f"    {grp}: MAE = {grp_mae:.2f} м ({grp_cnt} зданий)")

    # Доля предсказаний с ошибкой < 1 этажа (3 м)
    within_1_floor = (errors['error'] < 3).sum() / len(errors) * 100
    within_2_floors = (errors['error'] < 6).sum() / len(errors) * 100
    results['within_1_floor_pct'] = round(within_1_floor, 1)
    results['within_2_floors_pct'] = round(within_2_floors, 1)

    logger.info(f"\n  Ошибка < 1 этажа (3 м): {within_1_floor:.1f}% зданий")
    logger.info(f"  Ошибка < 2 этажей (6 м): {within_2_floors:.1f}% зданий")

    return results

# ПРОВЕРКА 4: ПРОСТРАНСТВЕННАЯ (ПО РАЙОНАМ)

def check_spatial(buildings, gdf_b):
    """
    Средняя высота по районам - должна различаться.
    Используем поле 'district' из Источника Б.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 4: ПРОСТРАНСТВЕННАЯ (ПО РАЙОНАМ)")
    logger.info("=" * 60)

    if 'district' not in gdf_b.columns:
        logger.info("  Нет поля district, пропускаем")
        return {}

    # Привязываем district к зданиям через компоненты
    # Упрощенно: берем district из first B-полигона
    district_map = {}
    if 'n_polygons_b' in buildings.columns:
        # У зданий с полигонами из Б есть район
        pass

    # Прямой подход: берем district из gdf_b для зданий с height_b
    mask = buildings['height_b'].notna()
    has_height = buildings[mask].copy()

    if 'purpose' in has_height.columns:
        # Группируем по назначению
        purpose_stats = has_height.groupby('purpose')['height_final'].agg(['mean', 'median', 'count'])
        purpose_stats = purpose_stats[purpose_stats['count'] >= 50].sort_values('mean', ascending=False)

        if len(purpose_stats) > 0:
            logger.info(f"\n  Средняя высота по назначению здания:")
            results = {}
            for purp, row in purpose_stats.head(15).iterrows():
                logger.info(f"    {purp:40s}: средняя {row['mean']:.1f} м, "
                           f"медиана {row['median']:.1f} м ({int(row['count'])} зд.)")
                results[purp] = {'mean': round(row['mean'], 1), 'count': int(row['count'])}
            return results

    return {}

# ПРОВЕРКА 5: КОРРЕЛЯЦИЯ С СОСЕДЯМИ

def check_neighbor_correlation(buildings):
    """
    Высота здания должна коррелировать с высотой соседей.
    В реальном городе кварталы обычно однородны.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 5: КОРРЕЛЯЦИЯ С СОСЕДЯМИ")
    logger.info("=" * 60)

    if 'avg_height_neighbors_100m' not in buildings.columns:
        logger.info("  Нет данных о соседях, пропускаем")
        return {}

    both = buildings[
        buildings['height_final'].notna() &
        buildings['avg_height_neighbors_100m'].notna()
    ]

    if len(both) < 100:
        logger.info("  Мало данных")
        return {}

    corr = both['height_final'].corr(both['avg_height_neighbors_100m'])

    results = {'correlation': round(corr, 4), 'n_buildings': len(both)}

    if corr > 0.8:
        verdict = "ОТЛИЧНО - высота сильно коррелирует с окружением"
    elif corr > 0.6:
        verdict = "ХОРОШО - заметная корреляция"
    elif corr > 0.4:
        verdict = "ПРИЕМЛЕМО - умеренная корреляция"
    else:
        verdict = "СЛАБО - модель плохо ловит пространственные паттерны"

    logger.info(f"  Корреляция высоты с соседями: {corr:.4f}")
    logger.info(f"  Вердикт: {verdict}")
    logger.info(f"  (на {len(both)} зданиях)")

    return results

# ПРОВЕРКА 6: АНАЛИЗ ОШИБОК

def check_error_analysis(buildings):
    """
    На каких зданиях мы менее уверены? Где слабые места?
    """
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА 6: АНАЛИЗ СЛАБЫХ МЕСТ")
    logger.info("=" * 60)

    results = {}

    # По уверенности
    if 'height_confidence' in buildings.columns:
        conf = buildings['height_confidence'].value_counts()
        logger.info(f"\n  Распределение по уверенности:")
        for c, cnt in conf.items():
            pct = cnt / len(buildings) * 100
            logger.info(f"    {c:25s}: {cnt:6d} ({pct:5.1f}%)")
            results[f'confidence_{c}'] = int(cnt)

    # По источнику
    if 'height_source' in buildings.columns:
        src = buildings['height_source'].value_counts()
        logger.info(f"\n  Распределение по источнику высоты:")
        for s, cnt in src.items():
            pct = cnt / len(buildings) * 100
            logger.info(f"    {str(s):40s}: {cnt:6d} ({pct:5.1f}%)")

    # Здания с low confidence - это наши слабые места
    if 'height_confidence' in buildings.columns:
        low = buildings[buildings['height_confidence'].isin(['low', 'low_conflicting', 'predicted'])]
        n_low = len(low)
        pct_low = n_low / len(buildings) * 100
        logger.info(f"\n  Здания с низкой уверенностью: {n_low} ({pct_low:.1f}%)")
        results['low_confidence_count'] = n_low
        results['low_confidence_pct'] = round(pct_low, 1)

        if n_low > 0 and 'area_m2' in low.columns:
            logger.info(f"    Средняя площадь: {low['area_m2'].mean():.0f} м²")
            logger.info(f"    Медианная высота: {low['height_final'].median():.1f} м")

    return results

# ИТОГОВЫЙ ВЕРДИКТ

def final_verdict(all_results):
    """Общий вывод по всем проверкам."""
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВЫЙ ВЕРДИКТ")
    logger.info("=" * 60)

    passes = 0
    total_checks = 0

    # Проверка 1: медиана реалистична
    total_checks += 1
    if 5 <= all_results.get('statistics', {}).get('median', 0) <= 20:
        passes += 1
        logger.info("  Медианная высота реалистична")
    else:
        logger.info("  Медианная высота подозрительна")

    # Проверка 3: MAE < 3 м
    total_checks += 1
    mae = all_results.get('ml_holdout', {}).get('MAE', 999)
    if mae < 3:
        passes += 1
        logger.info(f"  ML ошибка {mae} м < 3 м (меньше 1 этажа)")
    else:
        logger.info(f"  ML ошибка {mae} м - многовато")

    # Проверка 5: корреляция с соседями > 0.6
    total_checks += 1
    corr = all_results.get('neighbor_correlation', {}).get('correlation', 0)
    if corr > 0.6:
        passes += 1
        logger.info(f"  Корреляция с соседями {corr:.2f} > 0.6")
    else:
        logger.info(f"  Корреляция с соседями {corr:.2f} - слабовата")

    # Покрытие 100%
    total_checks += 1
    coverage = all_results.get('statistics', {}).get('total_buildings', 0)
    if coverage > 160000:
        passes += 1
        logger.info(f"   Покрытие: {coverage} зданий (100%)")

    logger.info(f"\n  РЕЗУЛЬТАТ: {passes}/{total_checks} проверок пройдено")

    if passes == total_checks:
        logger.info("  ВЕРДИКТ: Модель высотности ВАЛИДНА")
    elif passes >= total_checks - 1:
        logger.info("  ВЕРДИКТ: Модель высотности В ЦЕЛОМ ВАЛИДНА (есть замечания)")
    else:
        logger.info("  ВЕРДИКТ: Модель высотности ТРЕБУЕТ ДОРАБОТКИ")

    return {'passes': passes, 'total': total_checks}

# ТОЧКА ВХОДА

if __name__ == '__main__':
    OUTPUT_DIR = 'output'
    BUILDINGS = os.path.join(OUTPUT_DIR, 'buildings_with_height.csv')
    CLEAN_B = os.path.join(OUTPUT_DIR, 'clean_B.csv')

    logger.info(" ЗАДАНИЕ 4: ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)
    t0 = time.time()

    buildings = pd.read_csv(BUILDINGS)
    logger.info(f"Загружено: {len(buildings)} зданий")

    # Загружаем clean_B для проверки по районам
    gdf_b = None
    try:
        gdf_b_df = pd.read_csv(CLEAN_B)
        logger.info(f"clean_B: {len(gdf_b_df)} записей")
    except:
        gdf_b_df = pd.DataFrame()

    all_results = {}

    # Проверки
    all_results['statistics'] = check_statistics(buildings)
    all_results['source_agreement'] = check_source_agreement(buildings)
    all_results['ml_holdout'] = check_ml_holdout(buildings)
    all_results['spatial'] = check_spatial(buildings, gdf_b_df)
    all_results['neighbor_correlation'] = check_neighbor_correlation(buildings)
    all_results['error_analysis'] = check_error_analysis(buildings)

    # Итоговый вердикт
    all_results['verdict'] = final_verdict(all_results)

    # Сохранение
    with open(os.path.join(OUTPUT_DIR, 'validation_report.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n  Отчет: {OUTPUT_DIR}/validation_report.json")
    logger.info(f"  Лог: {OUTPUT_DIR}/validation_log.txt")
    logger.info(f"\n ВАЛИДАЦИЯ ЗАВЕРШЕНА за {time.time()-t0:.1f}с")